# -*- coding: utf-8 -*-
"""
GRU 序列回归（向量分量 + 幅值/方向头 + 稀疏门控 + 两阶段训练）
----------------------------------------------------------------
- 数据：按文件切分 Train/Val，z-score 仅用训练集统计
- 预训练：仅使用“强力”样本（|F| > high_th_abs）
- 微调：全部样本 + 动态门控（目标非零占比 target_pos），混合强力 batch
- 损失：分量（signed-log + log-cosh）+ 幅值（log-cosh）+ 方向（余弦）+ 欠估惩罚/稀疏惩罚
- 评估：R²(各分量/|F|)、Pearson r、MAE/RMSE、非零比例、|F|线性标定
- 稳定性：nan_to_num 全面兜底 + step 级保险丝（回滚到 EMA、降学习率）+ 评估前清洗
"""

import os
import glob
import math
import time
import random
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 环境修正（Windows OpenMP/MKL 冲突）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

# ======================
# 配置
# ======================
DATA_DIR   = "nn_input_data_all"   # CSV 目录
GLOB_PAT   = "*.csv"

FEATURES = [
    "d", "r0", "h0",
    "pair_0_1_qrel_x", "pair_0_1_qrel_y", "pair_0_1_qrel_z", "pair_0_1_qrel_w",
]
TARGET_COLS = ["U_int_Fx", "U_int_Fy", "U_int_Fz"]

# 序列
SEQ_LEN    = 15
SEQ_STRIDE = 1

# 训练
BATCH_SIZE       = 1024
EPOCHS           = 60
PRETRAIN_EPOCHS  = 8       # 预训练 epoch 数
LR               = 1e-4
WEIGHT_DECAY     = 1e-4
CLIP_GRAD        = 1.0
SEED             = 2025
device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 目标缩放：训练时按 p95(|F|) 缩放到 ~1（自动用训练集统计）
Y_P95_SCALE = None  # None 表示自动用训练集 p95

# 阈值/分位参数
ZERO_EPS          = 1e-8        # 绝对零判断的极小值
NZ_THRESHOLD_ABS  = None        # None=按分位估计
NZ_QUANTILE       = 0.10        # 真非零的 p10（未直接使用，保留占位）
HIGHF_FIXED       = None        # 强力阈值固定值；None=按分位
HIGHF_QUANTILE    = 0.98        # 全体的高分位
TARGET_POS_FOR_THRESH = 0.08    # 目标：全体样本中约 8% 判为非零
PHYS_NOISE_MIN    = 1e-6        # 物理噪声下限，避免阈值过低
HIGHF_Q           = 0.995       # compute_thresholds_from_all 中的强力高分位

# 欠估/方向损失开关
BETA_SMOOTHL1        = 0.1  #（保留占位，不直接使用）
USE_DIR_LOSS         = True
DIR_LOSS_WEIGHT      = 0.5
USE_UNDER_PENALTY    = True
UNDER_PENALTY_WEIGHT = 1.0

# 早停
EARLYSTOP_PATIENCE = 20

# 输出
SAVE_DIR_BASE = "./artifacts_vec"

# 可选：硬平衡 0/非0 （当前默认关闭）
BALANCE_USE        = True
BALANCE_CHECK_COLS = ["U_int", "U_int_Fx", "U_int_Fy", "U_int_Fz", "U_int_Fmag"]
BALANCE_EPS        = 0.0
BALANCE_RANDOM_OFFSET = 0
BALANCE_BY_GROUP   = False
TIME_COL           = None

# ======================
# 工具函数
# ======================
def set_seed(s: int = SEED) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def r2_score_np(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot


def vec_mag(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip((a * a).sum(axis=-1), 1e-18, None))


def signed_log(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # 线性域 → 对称对数域：保留符号，压缩长尾
    return torch.sign(x) * torch.log1p(torch.abs(x) / eps)


def log_cosh(x: torch.Tensor) -> torch.Tensor:
    # 数值稳定版 log(cosh(x))
    x = x.abs()
    return x + F.softplus(-2.0 * x) - math.log(2.0)


def fix_quat_block(df: pd.DataFrame, prefix: str) -> None:
    """规整四元数列：归一化、强制 w>=0、异常替换为单位四元数"""
    cols = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z", f"{prefix}_w"]
    q = df[cols].astype(float).to_numpy()
    n = np.linalg.norm(q, axis=1, keepdims=True)

    bad = (n.squeeze() < 1e-9) | ~np.isfinite(n.squeeze())
    qn = np.ones_like(q)
    qn[:, 3] = 1.0  # 单位四元数
    good = ~bad
    qn[good] = q[good] / (n[good] + 1e-12)
    flip = qn[:, 3] < 0
    qn[flip] = -qn[flip]

    df.loc[:, cols] = qn
    print(f"[QFIX] {prefix}: flipped={int(flip.sum())}, to_I={int(bad.sum())}")


def load_and_balance_csv(data_dir: str, time_col: Optional[str] = TIME_COL) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, GLOB_PAT)))
    assert files, f"未找到数据：{data_dir}/{GLOB_PAT}"

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["__file__"] = os.path.basename(f)
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    # 排序键
    if (time_col is not None) and (time_col in all_df.columns):
        all_df[time_col] = pd.to_datetime(all_df[time_col], errors="coerce")
        use_time = all_df[time_col].notna().any()
    else:
        use_time = False
    if use_time:
        sort_key = [time_col]
    else:
        all_df["__row__"] = np.arange(len(all_df))
        sort_key = ["__row__"]

    all_df = all_df.sort_values(sort_key, kind="mergesort").reset_index(drop=True)

    # 判零
    miss = [c for c in BALANCE_CHECK_COLS if c not in all_df.columns]
    if miss:
        raise KeyError(f"缺少用于判零的列: {miss}")
    is_all_zero = all_df[BALANCE_CHECK_COLS].abs().le(BALANCE_EPS).all(axis=1)

    def _balance_pair(df_sub: pd.DataFrame) -> pd.DataFrame:
        df_zero = df_sub.loc[is_all_zero[df_sub.index]].copy()
        df_nonz = df_sub.loc[~is_all_zero[df_sub.index]].copy()
        Nz, Nnz = len(df_zero), len(df_nonz)

        if Nnz == 0 or Nz == 0:
            df_zero_sys = df_zero.iloc[0:0].copy()
        else:
            stride = max(1, Nz // Nnz)
            start = BALANCE_RANDOM_OFFSET % stride
            take = np.arange(start, Nz, stride)
            if len(take) >= Nnz:
                take = take[:Nnz]
            df_zero_sys = df_zero.iloc[take].copy()

        out = pd.concat([df_nonz, df_zero_sys], ignore_index=True)
        out = out.sort_values(sort_key, kind="mergesort").reset_index(drop=True)
        print(f"[BAL] 非0:{len(df_nonz)} 全0:{Nz} -> 抽样全0:{len(df_zero_sys)} 合并:{len(out)}")
        return out

    if BALANCE_BY_GROUP:
        parts = [ _balance_pair(g) for _, g in all_df.groupby("__file__", sort=False) ]
        bal = pd.concat(parts, ignore_index=True)
    else:
        bal = _balance_pair(all_df)

    fix_quat_block(bal, "pair_0_1_qrel")

    need_cols = FEATURES + TARGET_COLS + ["__file__"]
    miss2 = [c for c in need_cols if c not in bal.columns]
    if miss2:
        raise KeyError(f"缺少训练所需列: {miss2}")

    sub = bal[need_cols].copy()
    print(f"[BAL] 最终样本数: {len(sub)}（全局）")
    return sub


def load_all_csv(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, GLOB_PAT)))
    assert files, f"未找到数据：{data_dir}/{GLOB_PAT}"

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        fix_quat_block(df, "pair_0_1_qrel")
        sub = df[FEATURES + TARGET_COLS].copy()
        sub["__file__"] = os.path.basename(f)
        dfs.append(sub)

    all_df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] 读取 {len(files)} 个CSV, 共 {len(all_df)} 行")
    return all_df


def split_by_file(data: pd.DataFrame, val_ratio_files: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(data["__file__"].unique())
    n_val = max(1, int(round(len(files) * val_ratio_files)))
    val_files = files[-n_val:]
    tr = data[~data["__file__"].isin(val_files)].reset_index(drop=True)
    va = data[data["__file__"].isin(val_files)].reset_index(drop=True)
    print(f"[INFO] Train rows={len(tr)} Val rows={len(va)} | files={len(files)-n_val}/{n_val}")
    return tr, va


def build_sequences_df(
    df: pd.DataFrame, seq_len: int, stride: int,
    use_cols: list, target_cols: list
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    outs_X, outs_Y = [], []
    for _, g in df.groupby("__file__", sort=False):
        X = g[use_cols].to_numpy(dtype=np.float32)
        Y = g[target_cols].to_numpy(dtype=np.float32)
        T = len(g)
        if T < seq_len:
            continue
        for st in range(0, T - seq_len + 1, stride):
            ed = st + seq_len
            outs_X.append(X[st:ed])
            outs_Y.append(Y[ed - 1])
    if not outs_X:
        return None
    return np.stack(outs_X), np.stack(outs_Y)


def balance_sequences(X: np.ndarray, Y: np.ndarray, nz_th_scaled: float, epoch: int, max_ratio: int = 10):
    mag = np.linalg.norm(Y, axis=1)  # (N,)
    pos_idx = np.where(mag > nz_th_scaled)[0]     # 非零
    neg_idx = np.where(mag <= nz_th_scaled)[0]    # 近零

    # 动态 0:非0 采样比
    if epoch < 3:
        zero_per_pos = 0.5
    elif epoch < 8:
        zero_per_pos = 1.0
    elif epoch < 15:
        zero_per_pos = 1.5
    else:
        zero_per_pos = 2.0
    zero_per_pos = max(min(zero_per_pos, max_ratio), 1.0 / max_ratio)

    n_pos = len(pos_idx)
    if n_pos == 0:
        keep = neg_idx
    else:
        n_neg_target = int(round(n_pos * zero_per_pos))
        if len(neg_idx) >= n_neg_target:
            neg_sample = np.random.choice(neg_idx, size=n_neg_target, replace=False)
        else:
            neg_sample = np.random.choice(neg_idx, size=n_neg_target, replace=True)
        keep = np.concatenate([pos_idx, neg_sample])

    np.random.shuffle(keep)
    return X[keep], Y[keep]


def bce_focal_elementwise(logits: torch.Tensor, targets: torch.Tensor,
                          alpha: float = 0.5, gamma: float = 1.5, eps: float = 1e-8) -> torch.Tensor:
    """逐元素 Focal BCE，不做 reduce。"""
    # 兜底
    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)

    if logits.dim() == 1:
        logits = logits.unsqueeze(-1)
    if targets.dim() == 1:
        targets = targets.unsqueeze(-1)

    prob = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    pt = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = - alpha_t * ((1 - pt) ** gamma) * (
        targets * torch.log(prob) + (1 - targets) * torch.log(1 - prob)
    )
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    return loss


# 长尾：对称 log 变换
EPS_LOG = 1e-6

def to_log_domain(Y: np.ndarray) -> np.ndarray:
    return np.sign(Y) * np.log1p(np.abs(Y) / EPS_LOG)

def from_log_domain(Y_log: np.ndarray) -> np.ndarray:
    return np.sign(Y_log) * (np.expm1(np.abs(Y_log)) * EPS_LOG)


# ======================
# 模型 & 损失
# ======================
class GRUReg(nn.Module):
    """
    V2: 输入嵌入 + 2层BiGRU + 注意力池化 + 残差共享头
    - in_dim -> embed_dim (线性+SiLU+LN)
    - BiGRU( embed_dim, hidden=128, num_layers=2, dropout=0.1 )
    - Attn Pooling: 用可学习query对时间维做加权求和（比取最后一步更稳）
    - Shared MLP(256->256) + Residual
    - Heads: mag(softplus)、dir(normalize)、nz_logit
    """
    def __init__(self, in_dim: int, embed_dim: int = 64, hidden_per_dir: int = 128,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.SiLU(),
            nn.LayerNorm(embed_dim),
        )
        self.bigru = nn.GRU(
            embed_dim, hidden_per_dir,
            num_layers=layers, batch_first=True,
            dropout=(dropout if layers > 1 else 0.0),
            bidirectional=True
        )
        feat_dim = hidden_per_dir * 2  # 双向拼接 -> 256
        # 可学习查询向量用于注意力池化
        self.query = nn.Parameter(torch.randn(1, 1, feat_dim))
        self.attn_ln = nn.LayerNorm(feat_dim)

        # 共享残差 MLP
        self.shared_fc1 = nn.Linear(feat_dim, feat_dim)
        self.shared_fc2 = nn.Linear(feat_dim, feat_dim)
        self.shared_drop = nn.Dropout(dropout)
        self.shared_act = nn.ReLU(inplace=True)
        self.shared_ln  = nn.LayerNorm(feat_dim)

        # 三个头
        self.head_mag = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, 1),
        )
        self.head_dir = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, 3),
        )
        self.head_nz = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, 1),
        )
        self._last_nz_logit = None

        # 简单初始化：避免早期饱和
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def attn_pool(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B,T,C)  BiGRU 输出
        返回: (B,C)  注意力聚合后的句向量
        """
        B, T, C = h.shape
        q = self.query.expand(B, -1, -1)              # (B,1,C)
        # 缩放点积注意力
        scores = torch.matmul(q, h.transpose(1, 2)) / math.sqrt(C)  # (B,1,T)
        attn = torch.softmax(scores, dim=-1)          # (B,1,T)
        pooled = torch.matmul(attn, h).squeeze(1)     # (B,C)
        return self.attn_ln(pooled)

    def shared_block(self, x: torch.Tensor) -> torch.Tensor:
        # 残差 MLP：LN -> FC -> ReLU -> Drop -> FC -> 残差 -> LN
        z = self.shared_ln(x)
        z = self.shared_fc1(z)
        z = self.shared_act(z)
        z = self.shared_drop(z)
        z = self.shared_fc2(z)
        return x + z

    def forward(self, x: torch.Tensor):
        # x: (B,T,in_dim)
        x = self.embed(x)                # (B,T,E)
        h, _ = self.bigru(x)             # (B,T,2H)
        s = self.attn_pool(h)            # (B,2H)
        s = self.shared_block(s)         # (B,2H)

        mag_raw  = self.head_mag(s)      # (B,1)
        dir_raw  = self.head_dir(s)      # (B,3)
        nz_logit = self.head_nz(s)       # (B,1)
        self._last_nz_logit = nz_logit

        # 数值兜底
        mag_raw  = torch.nan_to_num(mag_raw,  nan=0.0, posinf=0.0, neginf=0.0)
        dir_raw  = torch.nan_to_num(dir_raw,  nan=0.0, posinf=0.0, neginf=0.0)
        nz_logit = torch.nan_to_num(nz_logit, nan=0.0, posinf=0.0, neginf=0.0)

        mag   = F.softplus(mag_raw)
        direc = F.normalize(dir_raw, dim=-1)
        vec   = mag * direc

        vec    = torch.nan_to_num(vec,    nan=0.0, posinf=0.0, neginf=0.0)
        mag    = torch.nan_to_num(mag,    nan=0.0, posinf=0.0, neginf=0.0)
        direc  = torch.nan_to_num(direc,  nan=0.0, posinf=0.0, neginf=0.0)
        self._last_nz_logit = torch.nan_to_num(nz_logit, nan=0.0, posinf=0.0, neginf=0.0)

        return vec, mag, direc



class SeqDS(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])


def loss_fn(vec_pred: torch.Tensor, mag_pred: torch.Tensor, direc_pred: torch.Tensor, y_true: torch.Tensor,
            nz_th_scaled: float, high_th_scaled: float,
            weight_Fx: float = 1.0, weight_Fy: float = 1.8, weight_Fz: float = 1.8,
            w_mag: float = 1.0, w_dir: float = 0.0, eps: float = 1e-6) -> torch.Tensor:
    """
    组合损失（强化版）：
    - 分量：对称 log + log-cosh，并对非零/硬非零加权
    - 幅值：log-cosh，强力样本加权
    - 方向：仅在“硬非零”上计算
    - 稀疏/欠估：零样本抑制 + 强样本欠估惩罚
    - 推力：正样本幅值越过阈值的下界铰链
    - 矩匹配：批内 |F| 的均值/方差与真值对齐（轻度正则）
    """
    # --- 数值兜底 ---
    vec_pred   = torch.nan_to_num(vec_pred,   nan=0.0, posinf=0.0, neginf=0.0)
    mag_pred   = torch.nan_to_num(mag_pred,   nan=0.0, posinf=0.0, neginf=0.0)
    direc_pred = torch.nan_to_num(direc_pred, nan=0.0, posinf=0.0, neginf=0.0)
    y_true     = torch.nan_to_num(y_true,     nan=0.0, posinf=0.0, neginf=0.0)

    # --- 掩码 ---
    mag_true = torch.linalg.vector_norm(y_true, dim=-1, keepdim=True)  # (B,1) 线性域
    nz_mask  = (mag_true > nz_th_scaled).float()
    hi_mask  = (mag_true > high_th_scaled).float()
    hard_nz  = (mag_true > (1.5 * nz_th_scaled)).float()               # 远离阈值的“硬非零”

    # --- 幅值（log 域）---
    cosh_scale = 0.6  # 缩放 log-cosh 的残差尺度，增大大残差的相对梯度
    log_mag_pred = torch.log1p(mag_pred.clamp_min(0.0) + eps)
    log_mag_true = torch.log1p(mag_true.clamp_min(0.0) + eps)
    mag_diff = (log_mag_pred - log_mag_true) / cosh_scale
    mag_loss = log_cosh(mag_diff)
    mag_loss = mag_loss * (1.0 + 1.0 * hi_mask)                  # 强力样本加权
    mag_loss = (mag_loss * (0.1 + 0.9 * nz_mask)).mean()         # 近零样本减权

    # --- 分量（对称 log + log-cosh）---
    log_vec_pred = signed_log(vec_pred, eps=eps)
    log_vec_true = signed_log(y_true,   eps=eps)
    comp_diff = (log_vec_pred - log_vec_true) / cosh_scale
    comp_loss_raw = log_cosh(comp_diff)                           # (B,3)
    w = torch.tensor([weight_Fx, weight_Fy, weight_Fz],
                     device=vec_pred.device, dtype=vec_pred.dtype).view(1, 3)
    # 非零强、硬非零更强、零极弱
    comp_w = 0.05 + 0.45 * nz_mask + 0.50 * hard_nz               # (B,1)
    comp_loss = (comp_loss_raw * w * comp_w).mean()

    # --- 方向（仅硬非零）---
    if w_dir > 0.0:
        direc_pred_n = F.normalize(direc_pred, dim=-1)
        direc_true_n = F.normalize(y_true,    dim=-1).clamp_min(1e-12)
        cos_sim = (direc_pred_n * direc_true_n).sum(dim=-1, keepdim=True)
        dir_loss = (0.5 * (1.0 - cos_sim)) * hard_nz
        dir_loss = dir_loss.mean()
    else:
        dir_loss = torch.tensor(0.0, device=vec_pred.device)

    # --- 稀疏 / 欠估 惩罚 ---
    margin = 1.2 * nz_th_scaled
    L_sparse = (torch.relu(mag_pred - margin) * (1.0 - nz_mask)).mean() * 0.8
    L_under  = (torch.relu((mag_true - mag_pred)) * hi_mask).mean() * 0.6

    # --- 正样本推力（把幅值推过阈值的下界）---
    alpha_push = 0.8
    L_push = (torch.relu(alpha_push * nz_th_scaled - mag_pred) * nz_mask).mean() * 0.4

    # --- 批内矩匹配（|F| 的均值/方差对齐）---
    with torch.no_grad():
        m_t = mag_true.mean()
        s_t = mag_true.std().clamp_min(1e-6)
    m_p = mag_pred.mean()
    s_p = mag_pred.std().clamp_min(1e-6)
    L_moment = ((m_p - m_t).abs() + (s_p - s_t).abs()) * 0.2

    total = (
        w_mag * mag_loss
        + comp_loss
        + w_dir * dir_loss
        + L_sparse + L_under
        + L_push + L_moment
    )
    return torch.nan_to_num(total, nan=0.0, posinf=1e6, neginf=1e6)

def linear_calibration(x_pred, y_true, nonneg: bool = False):
    """一维线性标定 y≈a*x+b（岭回归极小正则）"""
    x = np.asarray(x_pred).reshape(-1, 1)
    y = np.asarray(y_true).reshape(-1, 1)
    X = np.concatenate([x, np.ones_like(x)], axis=1)
    lam = 1e-8
    coef = np.linalg.solve(X.T @ X + lam * np.eye(2), X.T @ y)
    a, b = float(coef[0, 0]), float(coef[1, 0])

    def apply(z):
        out = a * z + b
        if nonneg:
            out = np.maximum(out, 0.0)
        return out

    return a, b, apply


def mixed_batches(train_loader, strong_loader=None, ratio_strong=0.2, steps=None):
    if steps is None:
        steps = len(train_loader)
    it_tr = iter(train_loader)
    it_str = iter(strong_loader) if strong_loader is not None else None
    for _ in range(steps):
        use_str = (it_str is not None) and (random.random() < float(ratio_strong))
        if use_str:
            try:
                xs, ys = next(it_str)
            except StopIteration:
                it_str = iter(strong_loader)
                try:
                    xs, ys = next(it_str)
                except StopIteration:
                    use_str = False
        if not use_str:
            try:
                xs, ys = next(it_tr)
            except StopIteration:
                it_tr = iter(train_loader)
                xs, ys = next(it_tr)
        yield xs, ys


def apply_gate(ep: int, nz_logit: torch.Tensor, mag: torch.Tensor, direc: torch.Tensor,
               nz_th_scaled: float, target_pos: float, temp: float = 0.75, dead_k: float = 1.0):
    """基于 nz_logit 的 Top-k 门控 + 死区抑制"""
    B = nz_logit.shape[0]
    nz_logit = nz_logit.view(B, 1)

    # Top-k 阈值（按目标占比）
    k = max(1, int(math.ceil(float(target_pos) * B)))
    tau_logit = torch.topk(nz_logit.flatten(), k, largest=True).values.min()

    # 直通估计：hard mask 的梯度用 soft mask 近似
    mask_hard = (nz_logit >= tau_logit).float()
    mask_soft = torch.sigmoid((nz_logit - tau_logit) / temp)
    mask = mask_hard.detach() + (mask_soft - mask_soft.detach())

    # 死区与退火
    if ep <= 3:
        base_mul = 0.3
    elif ep <= 8:
        base_mul = 0.3
    elif ep <= 15:
        base_mul = 0.25
    else:
        base_mul = 0.25

    deadzone = (dead_k * base_mul) * nz_th_scaled
    deadzone_vec = torch.where(
        (mask > 0.5).expand_as(mag),
        torch.zeros_like(mag),
        torch.full_like(mag, deadzone)
    )
    mag = torch.clamp(mag - deadzone_vec, min=0.0)

    direc = F.normalize(direc, dim=-1)
    vec = mag * direc
    # 兜底
    vec = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    mag = torch.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)

    return vec, mag, mask, tau_logit


# ======================
# 主流程
# ======================
def main():
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from scipy.stats import pearsonr

    set_seed()
    print(f"[INFO] device={device}")

    # 读数据
    if BALANCE_USE:
        print("[INFO] 使用硬平衡：全0 vs 非0（系统抽样）")
        df = load_and_balance_csv(DATA_DIR, time_col=TIME_COL)
    else:
        df = load_all_csv(DATA_DIR)

    # 按“文件”切分 Train/Val
    tr_df, va_df = split_by_file(df, val_ratio_files=0.2)

    # 仅用“训练集”的物理尺度 |F| 计算阈值/缩放
    Ytr_abs = tr_df[TARGET_COLS].to_numpy(dtype=np.float64)
    all_mag_train = vec_mag(Ytr_abs)

    # 仅用于统计打印（不参与阈值）
    Yall_abs = df[TARGET_COLS].to_numpy(dtype=np.float64)
    all_mag_all = vec_mag(Yall_abs)

    def compute_thresholds_from_all(
        all_mag: np.ndarray,
        target_pos: float = TARGET_POS_FOR_THRESH,
        phys_noise_min: float = PHYS_NOISE_MIN,
        highf_fixed: Optional[float] = HIGHF_FIXED,
        highf_q: float = HIGHF_Q
    ) -> Tuple[float, float, float]:
        mag = np.asarray(all_mag, dtype=np.float64)
        mag = mag[np.isfinite(mag)]
        if mag.size == 0:
            return 1e-12, 1e-12, 1e-12

        target_pos = float(np.clip(target_pos, 1e-3, 0.2))
        nz_th_abs = float(np.quantile(mag, 1.0 - target_pos))          # 非零阈值
        nz_th_abs = max(nz_th_abs, phys_noise_min)

        pos = mag[mag > phys_noise_min]
        p95_abs = float(np.quantile(pos, 0.90)) if pos.size > 0 else max(nz_th_abs, 1e-12)  # 训练缩放
        if highf_fixed is not None:
            high_th_abs = float(highf_fixed)
        else:
            high_th_abs = float(np.quantile(mag, highf_q))
        high_th_abs = float(max(high_th_abs, 1.5 * nz_th_abs))         # 保证强于非零阈值

        return nz_th_abs, p95_abs, high_th_abs

    nz_th_abs, p95_abs, high_th_abs = compute_thresholds_from_all(
        all_mag_train,
        target_pos=TARGET_POS_FOR_THRESH,
        phys_noise_min=PHYS_NOISE_MIN,
        highf_fixed=HIGHF_FIXED,
        highf_q=HIGHF_Q,
    )

    y_scale = p95_abs if (Y_P95_SCALE is None) else float(Y_P95_SCALE)
    nz_th_scaled   = nz_th_abs / y_scale
    high_th_scaled = high_th_abs / y_scale

    print(f"[THRESH] NZ_THRESHOLD(abs)={nz_th_abs:.3e}, HIGHF_FIXED(abs)={high_th_abs:.3e}")
    print(f"[SCALE→TRAIN] y_scale(p95 |F|)={y_scale:.3e} -> nz_th_scaled={nz_th_scaled:.3e}, high_th_scaled={high_th_scaled:.3e}")

    print(f"[STAT] 总样本数 = {len(all_mag_all)}")
    print(f"[STAT] |F| 最小 = {all_mag_all.min():.3e}, 最大 = {all_mag_all.max():.3e}")
    print(f"[STAT] |F| 均值 = {all_mag_all.mean():.3e}, 中位 = {np.median(all_mag_all):.3e}")
    nonzero_frac_all = float((all_mag_all > nz_th_abs).mean())
    print(f"[STAT] 非零样本比例(|F|>{nz_th_abs:.3e}) = {nonzero_frac_all:.3f}")

    strong_mask_tr = (all_mag_train > high_th_abs)
    strong_df = tr_df.loc[strong_mask_tr].reset_index(drop=True)
    strong_frac_all = float((all_mag_all > high_th_abs).mean())
    print(f"[STAT] “强力”阈值(abs) = {high_th_abs:.3e}, 全体占比 {strong_frac_all:.3f}")
    print(f"[PRE-TRAINING] 共 {len(strong_df)} 条强力样本用于预训练")

    # 标准化特征：用训练集均值/方差
    mu  = tr_df[FEATURES].mean().to_numpy()
    std = tr_df[FEATURES].std().to_numpy() + 1e-12

    def z(df_):
        X = df_[FEATURES].to_numpy()
        return (X - mu) / std

    trZ = tr_df.copy(); trZ[FEATURES] = z(tr_df)
    vaZ = va_df.copy(); vaZ[FEATURES] = z(va_df)

    # ========= 预训练（强力样本）=========
    print("[Stage1] 预训练（强力样本，使用与训练/验证一致的 z-score）")
    strongZ = strong_df.copy(); strongZ[FEATURES] = z(strong_df)
    seq_s = build_sequences_df(strongZ, SEQ_LEN, SEQ_STRIDE, FEATURES, TARGET_COLS)
    if seq_s is None:
        raise RuntimeError("强力样本不足构建序列")
    X_s, Y_s_abs = seq_s
    Y_s = Y_s_abs / y_scale

    loader_s = DataLoader(SeqDS(X_s, Y_s), batch_size=BATCH_SIZE, shuffle=True)

    model = GRUReg(in_dim=len(FEATURES), embed_dim=64, hidden_per_dir=128, layers=2, dropout=0.1).to(device)

    # 先验偏置：按稀疏率设定分类头 bias
    def set_logit_bias(module: nn.Module, p0: float = 0.05):
        b = math.log(p0 / (1 - p0))
        for m in module.modules():
            if isinstance(m, nn.Linear) and m.out_features == 1 and m.bias is not None:
                with torch.no_grad():
                    m.bias.fill_((b))

    set_logit_bias(model, p0=0.047)

    # 冻结 GRU 仅训练 heads
    for p in model.bigru.parameters():
        p.requires_grad = False

    opt_s = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=LR, weight_decay=WEIGHT_DECAY)

    for ep in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        tot = 0.0; n = 0
        for xb, yb in loader_s:
            xb = xb.to(device); yb = yb.to(device)
            opt_s.zero_grad(set_to_none=True)
            vec, mag, direc = model(xb)
            loss = loss_fn(
                vec, mag, direc, yb,
                nz_th_scaled=0.0, high_th_scaled=0.0,
                weight_Fx=1.0, weight_Fy=3.0, weight_Fz=3.0,
                w_mag=1.0, w_dir=2.0
            )
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if CLIP_GRAD > 0:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            opt_s.step()
            tot += float(loss.item()) * xb.size(0); n += xb.size(0)
        print(f"[PreEp{ep:02d}] Loss={tot / max(1, n):.5f}")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(SAVE_DIR_BASE) / f"GRU_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "pretrained_model.pt")

    # ========= 微调（全部样本）=========
    print("[Stage2] 微调（全部样本）")
    seq_tr = build_sequences_df(trZ, SEQ_LEN, SEQ_STRIDE, FEATURES, TARGET_COLS)
    seq_va = build_sequences_df(vaZ, SEQ_LEN, SEQ_STRIDE, FEATURES, TARGET_COLS)
    assert seq_tr is not None and seq_va is not None, "序列长度不足"

    Xtr, Ytr_abs = seq_tr
    Xva, Yva_abs = seq_va
    Ytr = Ytr_abs / p95_abs
    Yva = Yva_abs / p95_abs

    # 注意：此处按 p95_abs 做缩放（和上面 y_scale 一致）
    nz_th_scaled   = nz_th_abs   / p95_abs
    high_th_scaled = high_th_abs / p95_abs
    print(f"[THRESH] NZ_THRESHOLD(abs)={nz_th_abs:.3e}, HIGHF_FIXED(abs)={high_th_abs:.3e}")
    print(f"[SCALE→TRAIN] y_scale(p95 |F|)={p95_abs:.3e} -> nz_th_scaled={nz_th_scaled:.3e}, high_th_scaled={high_th_scaled:.3e}")

    # 首轮平衡 & Loader
    Xtr_bal, Ytr_bal = balance_sequences(Xtr, Ytr, nz_th_scaled, epoch=0, max_ratio=5)
    train_loader = DataLoader(SeqDS(Xtr_bal, Ytr_bal), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(SeqDS(Xva, Yva),       batch_size=BATCH_SIZE, shuffle=False)

    # 强力样本 Loader（备混合）
    strongZ = strong_df.copy()
    strongZ[FEATURES] = z(strong_df)
    seq_strong = build_sequences_df(strongZ, SEQ_LEN, SEQ_STRIDE, FEATURES, TARGET_COLS)
    if seq_strong is None:
        strong_loader = DataLoader(SeqDS(Xtr, Ytr), batch_size=max(8, BATCH_SIZE // 2), shuffle=True, drop_last=True)
    else:
        Xs, Ys_abs = seq_strong
        Ys = Ys_abs / p95_abs
        strong_loader = DataLoader(SeqDS(Xs, Ys), batch_size=max(8, BATCH_SIZE // 2), shuffle=True, drop_last=True)

    # 加载预训练权重，解冻全参
    model.load_state_dict(torch.load(out_dir / "pretrained_model.pt", map_location=device))
    for p in model.parameters():
        p.requires_grad = True

    # 分类头更大学习率
    base_lr = LR
    cls_params, other_params = [], []
    for n, p in model.named_parameters():
        if ('cls' in n) or ('nz' in n) or ('logit' in n):
            cls_params.append(p)
        else:
            other_params.append(p)

    opt = torch.optim.AdamW(
        [
            {"params": other_params, "lr": base_lr},
            {"params": cls_params,   "lr": base_lr * 5.0},
        ],
        weight_decay=1e-4
    )
    # 更低的 eta_min，后期更稳
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=base_lr * 0.02)

    # --- EMA 影子参数（用于回滚）---
    ema_decay = 0.995
    ema_shadow: Dict[str, torch.Tensor] = {k: v.detach().clone() for k, v in model.state_dict().items()}

    best = -1e9
    best_state = None
    no_imp = 0
    target_pos_ema = None
    alpha_ema = None
    ema_beta = 0.8

    for ep in range(1, EPOCHS + 1):
        Xtr_bal, Ytr_bal = balance_sequences(Xtr, Ytr, nz_th_scaled, epoch=ep, max_ratio=5)
        train_loader = DataLoader(SeqDS(Xtr_bal, Ytr_bal), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        ratio_strong = 0.60 if ep <= 6 else (0.45 if ep <= 12 else 0.30)

        model.train()
        tot = 0.0; n = 0

        # 目标非零占比（EMA→轻微上调）
        if target_pos_ema is None:
            target_pos = 0.08
        else:
            target_pos = float(np.clip(target_pos_ema * 1.10, 0.05, 0.12))

        use_gate_train = (ep >= 12)

        for step, (xb, yb) in enumerate(mixed_batches(train_loader, strong_loader, ratio_strong=ratio_strong)):
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)

            vec_raw, mag_raw, direc_raw = model(xb)
            direc_norm = F.normalize(direc_raw, dim=-1)

            nz_logit = getattr(model, "_last_nz_logit", None)
            if nz_logit is not None and nz_logit.dim() == 1:
                nz_logit = nz_logit.unsqueeze(-1)

            # 温度/死区退火
            if ep <= 3:
                temp, dead_k = 1.8, 0.6
            elif ep <= 10:
                temp, dead_k = 1.2, 0.6
            else:
                temp, dead_k = 1.0, 0.6

            # 门控
            if use_gate_train:
                if nz_logit is None:
                    raise RuntimeError("nz_logit not found; ensure model.forward stores _last_nz_logit.")
                vec, mag, mask, tau = apply_gate(
                    ep, nz_logit, mag_raw, direc_norm,
                    nz_th_scaled=nz_th_scaled,
                    target_pos=target_pos,
                    temp=temp, dead_k=dead_k
                )
            else:
                vec, mag = vec_raw, mag_raw
                mask = torch.ones_like(mag_raw)
                tau = torch.tensor(float('nan'), device=xb.device)

            # 二分类标签（真非零）
            y_mag_true = torch.linalg.vector_norm(yb, dim=-1, keepdim=True)
            y_nz = (y_mag_true > nz_th_scaled).float()

            # 可靠标注（远离阈值带）
            with torch.no_grad():
                band_lo = 0.5 * nz_th_scaled
                band_hi = 1.5 * nz_th_scaled
                reliable = ((y_mag_true < band_lo) | (y_mag_true > band_hi)).float()

            # 分类损失（Focal BCE），EMA 动态 alpha
            if use_gate_train:
                with torch.no_grad():
                    p_pos_batch = y_nz.mean().clamp(1e-4, 1 - 1e-4).item()
                    alpha_ema = (1.0 - p_pos_batch) if alpha_ema is None else (0.9 * alpha_ema + 0.1 * (1.0 - p_pos_batch))
                alpha_pos = float(np.clip(alpha_ema, 0.7, 0.98))
                cls_element = bce_focal_elementwise(nz_logit, y_nz, alpha=alpha_pos, gamma=2.0)
                reliable_sum = reliable.sum().clamp_min(1.0)
                loss_cls = (cls_element * reliable).sum() / reliable_sum
                # 分类项更保守，降低后期震荡
                if ep <= 6:
                    cls_weight = 0.0  # 纯回归预热
                elif ep <= 15:
                    cls_weight = 0.15
                else:
                    cls_weight = 0.10
            else:
                loss_cls = torch.tensor(0.0, device=xb.device)
                cls_weight = 0.0

            # 回归损失（门控后 & 全量）
            w_dir_curr = 0.15 if ep >= 15 else 0.0
            loss_reg_mask = loss_fn(
                vec, mag, direc_norm, yb,
                nz_th_scaled, high_th_scaled,
                weight_Fx=1.0, weight_Fy=1.8, weight_Fz=1.8,
                w_mag=1.0, w_dir=w_dir_curr
            )
            loss_reg_all = loss_fn(
                vec_raw, mag_raw, direc_norm, yb,
                nz_th_scaled, high_th_scaled,
                weight_Fx=1.0, weight_Fy=1.8, weight_Fz=1.8,
                w_mag=1.0, w_dir=0.0
            )

            if ep <= 6:
                w_all, w_mask = 0.7, 0.3
            else:
                t = min(1.0, max(0.0, (ep - 6) / 10.0))
                w_all = 0.2 + 0.3 * t
                w_mask = 1.0 - w_all

            loss_reg = w_mask * loss_reg_mask + w_all * loss_reg_all

            # 额外中心化约束，抑制 logit 漂移
            logit_center = (getattr(model, "_last_nz_logit") ** 2).mean() if use_gate_train else torch.tensor(0.0, device=xb.device)

            loss = loss_reg + cls_weight * loss_cls + 1e-4 * logit_center
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)

            # --- 保险丝：loss 非有限则跳过本 step（回滚到EMA + 降 LR）---
            if not torch.isfinite(loss):
                with torch.no_grad():
                    model.load_state_dict(ema_shadow, strict=True)
                for g in opt.param_groups:
                    g["lr"] = max(g["lr"] * 0.5, base_lr * 0.02)
                continue

            loss.backward()

            # --- 保险丝：梯度非有限则丢弃该 step（回滚 + 降 LR）---
            all_finite = True
            for p in model.parameters():
                if p.grad is not None and not torch.all(torch.isfinite(p.grad)):
                    all_finite = False
                    break
            if not all_finite:
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    model.load_state_dict(ema_shadow, strict=True)
                for g in opt.param_groups:
                    g["lr"] = max(g["lr"] * 0.5, base_lr * 0.02)
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            # --- 更新 EMA 影子参数 ---
            with torch.no_grad():
                cur_sd = model.state_dict()
                for k in cur_sd.keys():
                    ema_shadow[k].mul_(ema_decay).add_(cur_sd[k], alpha=1.0 - ema_decay)

            tot += float(loss.item()) * xb.size(0); n += xb.size(0)

        avg_loss = tot / max(1, n)
        sched.step()

        # ========== 验证 ==========
        model.eval()
        with torch.no_grad():
            taus = []
            YP_list = []
            nz_true_flags = []
            nz_pred_flags_mag = []
            nz_pred_flags_mask = []
            cls_probs = []

            use_gate_val = (ep >= 12)
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                vec_raw, mag_raw, direc = model(xb)
                direc = F.normalize(direc, dim=-1)
                logits = getattr(model, "_last_nz_logit")
                if logits.dim() == 1:
                    logits = logits.unsqueeze(-1)

                if use_gate_val:
                    vec_eval, mag_eval, mask, tau = apply_gate(
                        ep, logits, mag_raw, direc,
                        nz_th_scaled=nz_th_scaled,
                        target_pos=target_pos,
                        temp=0.5, dead_k=1.0
                    )
                else:
                    vec_eval, mag_eval = vec_raw, mag_raw
                    mask = torch.ones_like(mag_eval)
                    tau = torch.tensor(float('nan'), device=xb.device)

                taus.append(float(tau))
                # 反缩放到物理域前，先清 NaN
                vec_eval = torch.nan_to_num(vec_eval, nan=0.0, posinf=0.0, neginf=0.0)
                YP_list.append((vec_eval.detach().cpu().numpy() * p95_abs))

                y_mag_true = torch.linalg.vector_norm(yb, dim=-1, keepdim=True) * p95_abs
                y_nz_flag = (y_mag_true > nz_th_abs).float().cpu()
                y_mag_pred = torch.linalg.vector_norm(vec_eval, dim=-1, keepdim=True) * p95_abs
                y_pred_flag_mag = (y_mag_pred > nz_th_abs).float().cpu()
                y_pred_flag_mask = (mask > 0.5).float().cpu()

                nz_true_flags.append(y_nz_flag)
                nz_pred_flags_mag.append(y_pred_flag_mag)
                nz_pred_flags_mask.append(y_pred_flag_mask)
                cls_probs.append(torch.sigmoid(logits).detach().cpu())

        YP_abs = np.concatenate(YP_list, axis=0) if len(YP_list) else np.zeros((0, 3))

        nz_true = torch.cat(nz_true_flags).mean().item()
        nz_pred_mag = torch.cat(nz_pred_flags_mag).mean().item()
        nz_pred_mask = torch.cat(nz_pred_flags_mask).mean().item()
        nz_prob_mean = torch.cat(cls_probs).mean().item()
        tau_mean = np.mean(taus) if len(taus) else float('nan')

        print(f"[Ep{ep:03d}] Loss={avg_loss:.5f} | cls_prob(mean)={nz_prob_mean:.3f} | "
              f"NZ(true/mask/mag)={nz_true:.3f}/{nz_pred_mask:.3f}/{nz_pred_mag:.3f}")

        # EMA 追踪真实非零占比
        target_pos_ema = nz_true if target_pos_ema is None else (ema_beta * target_pos_ema + (1 - ema_beta) * nz_true)

        # 评价指标（反缩放后）
        magT  = vec_mag(Yva_abs)
        magP  = vec_mag(YP_abs)
        r2_fx = r2_score_np(Yva_abs[:, 0], YP_abs[:, 0])
        r2_fy = r2_score_np(Yva_abs[:, 1], YP_abs[:, 1])
        r2_fz = r2_score_np(Yva_abs[:, 2], YP_abs[:, 2])
        r2_all = r2_score_np(magT, magP)
        mask_nz = (magT > nz_th_abs)
        r2_nz   = r2_score_np(magT[mask_nz], magP[mask_nz]) if mask_nz.any() else float('nan')

        nz_true_ratio = float((magT > nz_th_abs).mean())
        nz_pred_ratio = float((magP > nz_th_abs).mean())

        print(f"[Ep{ep:03d}] R2(Fx,Fy,Fz)=({r2_fx:.3f},{r2_fy:.3f},{r2_fz:.3f}) | "
              f"|F| R2_all={r2_all:.3f} R2_nz={r2_nz:.3f} | NZ(true/pred)={nz_true_ratio:.3f}/{nz_pred_ratio:.3f}")

        # 早停：以 R2_nz 为准
        score = r2_nz
        if np.isfinite(score) and (score > best + 1e-5):
            best = score; no_imp = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= EARLYSTOP_PATIENCE:
                print(f"[EarlyStopping] stop at epoch {ep}")
                break

    # 还原最优
    if best_state is not None:
        model.load_state_dict(best_state)

    # 保存权重与 z-score
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mu":        mu,
            "std":       std,
            "features":  FEATURES,
            "y_scale":   p95_abs,
        },
        out_dir / "best_vecgru.pt"
    )

    # |F| 线性标定与指标前，确保预测是有限数
    YP_abs = np.nan_to_num(YP_abs, nan=0.0, posinf=0.0, neginf=0.0)
    magT = vec_mag(Yva_abs)
    magP = vec_mag(YP_abs)

    # 线性标定
    a_m, b_m, fn_mag = linear_calibration(magP, magT, nonneg=True)
    magP_cal = fn_mag(magP)
    r2_all_c = r2_score_np(magT, magP_cal)
    mask_nz = (magT > nz_th_abs)
    r2_nz_c = r2_score_np(magT[mask_nz], magP_cal[mask_nz]) if mask_nz.any() else float('nan')
    print(f"[CALIB] |F|_calib = max(0, {a_m:.6f} * |F|_pred + {b_m:.6f}) | R2_all_cal={r2_all_c:.3f} R2_nz_cal={r2_nz_c:.3f}")

    # 指标 CSV
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae_fx = mean_absolute_error(Yva_abs[:, 0], YP_abs[:, 0])
    mae_fy = mean_absolute_error(Yva_abs[:, 1], YP_abs[:, 1])
    mae_fz = mean_absolute_error(Yva_abs[:, 2], YP_abs[:, 2])
    rmse_fx = math.sqrt(mean_squared_error(Yva_abs[:, 0], YP_abs[:, 0]))
    rmse_fy = math.sqrt(mean_squared_error(Yva_abs[:, 1], YP_abs[:, 1]))
    rmse_fz = math.sqrt(mean_squared_error(Yva_abs[:, 2], YP_abs[:, 2]))
    mae_mag = mean_absolute_error(magT, magP)
    rmse_mag = math.sqrt(mean_squared_error(magT, magP))

    from scipy.stats import pearsonr
    r_fx, _  = pearsonr(Yva_abs[:, 0], YP_abs[:, 0])
    r_fy, _  = pearsonr(Yva_abs[:, 1], YP_abs[:, 1])
    r_fz, _  = pearsonr(Yva_abs[:, 2], YP_abs[:, 2])
    r_mag, _ = pearsonr(magT, magP)

    pd.DataFrame([{
        "R2_Fx": r2_fx, "R2_Fy": r2_fy, "R2_Fz": r2_fz,
        "Pearson_r_Fx": r_fx, "Pearson_r_Fy": r_fy, "Pearson_r_Fz": r_fz, "Pearson_r_|F|": r_mag,
        "MAE_Fx": mae_fx, "MAE_Fy": mae_fy, "MAE_Fz": mae_fz,
        "RMSE_Fx": rmse_fx, "RMSE_Fy": rmse_fy, "RMSE_Fz": rmse_fz,
        "R2_|F|_all_raw": r2_all, "R2_|F|_nz_raw": r2_nz,
        "MAE_|F|": mae_mag, "RMSE_|F|": rmse_mag,
        "R2_|F|_all_cal": r2_all_c, "R2_|F|_nz_cal": r2_nz_c,
        "NZ_true_ratio": float((magT > nz_th_abs).mean()),
        "NZ_pred_ratio": float((magP > nz_th_abs).mean()),
        "calib_|F|_a": a_m, "calib_|F|_b": b_m,
        "y_scale_p95": p95_abs, "highF_abs_value": high_th_abs
    }]).to_csv(out_dir / "val_metrics.csv", index=False)

    print("[DONE]")


if __name__ == "__main__":
    main()
