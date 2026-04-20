# -*- coding: utf-8 -*-
# ---------- OpenMP/MKL on Windows: 避免重复加载崩溃 ----------
import os

from sklearn.metrics import r2_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import glob, time, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ======================
# 配置
# ======================
DATA_DIR   = "nn_input_data_all"   # 你的 CSV 目录
GLOB_PAT   = "*.csv"

FEATURES = [
    "d","r0","h0",
    "pair_0_1_qrel_x","pair_0_1_qrel_y","pair_0_1_qrel_z","pair_0_1_qrel_w"
]
TARGET_COLS = ["U_int_Fx","U_int_Fy","U_int_Fz"]

# 序列
SEQ_LEN    = 15
SEQ_STRIDE = 1

# 训练
BATCH_SIZE       = 1024
EPOCHS           = 60
PRETRAIN_EPOCHS  = 8   # 预训练阶段 epoch 数
LR               = 1e-4
WEIGHT_DECAY     = 1e-4
CLIP_GRAD        = 1.0
SEED             = 2025
device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# —— 目标缩放：训练时按 p95(|F|) 缩放到 ~1，再在评估/画图时反缩放回来 ——
Y_P95_SCALE = None  # 设为 None 表示自动用训练集 p95

# ========= Threshold & scaling defaults =========
ZERO_EPS = 1e-8          # 小于此量级的 |F| 当作“伪零”，只在 >ZERO_EPS 上做分位统计
NZ_THRESHOLD_ABS = None  # None 表示用分位自适应（真非零分布）
NZ_QUANTILE = 0.10       # 非零阈值 = 真非零的 p10
HIGHF_FIXED = None       # 不再用固定强力阈值；None=用分位
HIGHF_QUANTILE = 0.98    # 强力阈值 = 真非零的 p95

# —— 损失：两种 loss 同时启用 + 欠估惩罚——
BETA_SMOOTHL1        = 0.1
USE_DIR_LOSS         = True
DIR_LOSS_WEIGHT      = 0.5

USE_UNDER_PENALTY    = True
UNDER_PENALTY_WEIGHT = 1.0

# —— 损失内样本加权
USE_LOSS_WEIGHT = False
W_NONZERO       = 2.0
W_HIGHF         = 4.0

# ---- gate/阈值计算相关新常量 ----
TARGET_POS_FOR_THRESH = 0.08   # 期望“全体样本”里约 3% 被视为非零
PHYS_NOISE_MIN       = 1e-6    # 物理噪声下限，避免阈值过低
HIGHF_Q              = 0.995   # 若没固定强力阈值，用全体的高分位

# 早停
EARLYSTOP_PATIENCE = 20

# 输出
SAVE_DIR_BASE = "./artifacts_vec"

BALANCE_USE        = True
BALANCE_CHECK_COLS = ["U_int", "U_int_Fx", "U_int_Fy", "U_int_Fz", "U_int_Fmag"]
BALANCE_EPS        = 0.0
BALANCE_RANDOM_OFFSET = 0
BALANCE_BY_GROUP   = False
TIME_COL           = None

# ======================
# 工具
# ======================
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def r2_score_np(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2) + 1e-12
    return 1.0 - ss_res/ss_tot

def vec_mag(a):
    return np.sqrt(np.clip((a*a).sum(axis=-1), 1e-18, None))

def signed_log(x, eps=1e-6):
    # 线性域 → 对称的对数域：保留符号，压缩长尾
    return torch.sign(x) * torch.log1p(torch.abs(x) / eps)

def log_cosh(x):
    x = x.abs()
    return x + F.softplus(-2.0 * x) - math.log(2.0)

def fix_quat_block(df: pd.DataFrame, prefix: str):
    cols = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z", f"{prefix}_w"]
    q = df[cols].astype(float).to_numpy()
    n = np.linalg.norm(q, axis=1, keepdims=True)
    bad = (n.squeeze()<1e-9) | ~np.isfinite(n.squeeze())
    qn = np.ones_like(q); qn[:,3]=1.0
    good = ~bad
    qn[good] = q[good] / (n[good] + 1e-12)
    flip = qn[:,3] < 0
    qn[flip] = -qn[flip]
    df.loc[:, cols] = qn
    print(f"[QFIX] {prefix}: flipped={int(flip.sum())}, to_I={int(bad.sum())}")

def load_and_balance_csv(data_dir, time_col=TIME_COL) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, GLOB_PAT)))
    assert files, f"未找到数据：{data_dir}/{GLOB_PAT}"
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["__file__"] = os.path.basename(f)
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    # 排序
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

    def _balance_pair(df_sub):
        df_zero  = df_sub.loc[ is_all_zero[df_sub.index] ].copy()
        df_nonz  = df_sub.loc[~is_all_zero[df_sub.index] ].copy()
        Nz, Nnz = len(df_zero), len(df_nonz)
        if Nnz == 0:
            df_zero_sys = df_zero.iloc[0:0].copy()
        elif Nz == 0:
            df_zero_sys = df_zero.iloc[0:0].copy()
        else:
            stride = max(1, Nz // Nnz)
            start  = BALANCE_RANDOM_OFFSET % stride
            take   = np.arange(start, Nz, stride)
            if len(take) >= Nnz:
                take = take[:Nnz]
            df_zero_sys = df_zero.iloc[take].copy()
        out = pd.concat([df_nonz, df_zero_sys], ignore_index=True)
        out = out.sort_values(sort_key, kind="mergesort").reset_index(drop=True)
        print(f"[BAL] 非0:{len(df_nonz)} 全0:{Nz} -> 抽样全0:{len(df_zero_sys)} 合并:{len(out)}")
        return out

    if BALANCE_BY_GROUP:
        parts = []
        for _, g in all_df.groupby("__file__", sort=False):
            parts.append(_balance_pair(g))
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

def load_all_csv(data_dir) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, GLOB_PAT)))
    assert files, f"未找到数据：{data_dir}/{GLOB_PAT}"
    dfs=[]
    for f in files:
        df = pd.read_csv(f)
        fix_quat_block(df, "pair_0_1_qrel")
        sub = df[FEATURES+TARGET_COLS].copy()
        sub["__file__"] = os.path.basename(f)
        dfs.append(sub)
    all_df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] 读取 {len(files)} 个CSV, 共 {len(all_df)} 行")
    return all_df

def split_by_file(data: pd.DataFrame, val_ratio_files=0.2):
    files = sorted(data["__file__"].unique())
    n_val  = max(1, int(round(len(files)*val_ratio_files)))
    val_files = files[-n_val:]
    tr = data[~data["__file__"].isin(val_files)].reset_index(drop=True)
    va = data[ data["__file__"].isin(val_files)].reset_index(drop=True)
    print(f"[INFO] Train rows={len(tr)} Val rows={len(va)} | files={len(files)-n_val}/{n_val}")
    return tr, va

def build_sequences_df(df, seq_len, stride, use_cols, target_cols):
    outs_X, outs_Y = [], []
    for _, g in df.groupby("__file__", sort=False):
        X = g[use_cols].to_numpy(dtype=np.float32)
        Y = g[target_cols].to_numpy(dtype=np.float32)
        T = len(g)
        if T < seq_len:
            continue
        for st in range(0, T-seq_len+1, stride):
            ed = st + seq_len
            outs_X.append(X[st:ed])
            outs_Y.append(Y[ed-1])
    if not outs_X:
        return None
    return np.stack(outs_X), np.stack(outs_Y)


def balance_sequences(X, Y, nz_th_scaled, epoch, max_ratio=10):
    """
    对序列级样本做动态的 undersample/oversample（按 非零:零 的目标比例）。
    这里的 ratio 定义为 zero_per_pos = N_zero / N_pos。
    前期：非零更密（学习‘有/无’）；后期：回到更接近真实的稀疏。
    """
    import numpy as np
    mag = np.linalg.norm(Y, axis=1)  # (N,)
    pos_idx = np.where(mag > nz_th_scaled)[0]     # 非零
    neg_idx = np.where(mag <= nz_th_scaled)[0]    # 近零

    # 序列级平衡（零:非零）
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
        # 兜底：全是零，随机取一批
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

# ---- 分类损失：Focal BCE（固定 alpha=0.10、gamma=2）----
def bce_focal_with_logits(logits, targets, alpha=0.10, gamma=2.0):
    # logits, targets shape: (B,1)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, p, 1.0 - p)
    alpha_t = torch.where(targets > 0.5, torch.tensor(alpha, device=logits.device, dtype=logits.dtype),
                          torch.tensor(1.0 - alpha, device=logits.device, dtype=logits.dtype))
    loss = (alpha_t * (1 - pt).pow(gamma) * bce).mean()
    return loss

# ==== 长尾数据专用：log-domain 转换函数 ====
EPS_LOG = 1e-6  # 避免 log(0) 溢出

def to_log_domain(Y: np.ndarray):
    """把原始物理尺度力值转为 log-domain"""
    return np.sign(Y) * np.log1p(np.abs(Y) / EPS_LOG)

def from_log_domain(Y_log: np.ndarray):
    """把 log-domain 预测值反变换回物理尺度"""
    return np.sign(Y_log) * (np.expm1(np.abs(Y_log)) * EPS_LOG)

# ======================
# 模型 & 损失
# ======================
class GRUReg(nn.Module):
    def __init__(self, in_dim, hidden=256, layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            in_dim, hidden, num_layers=layers, batch_first=True,
            dropout=(dropout if layers > 1 else 0.0)
        )
        self.norm   = nn.LayerNorm(hidden)
        self.shared = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        # 幅值与方向头
        self.head_mag = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1)   # raw magnitude
        )
        self.head_dir = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 3)   # raw direction
        )
        # 新增：非零分类头（logit，不做激活）
        self.head_nz = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1)   # nonzero logit
        )

        # 存最近一次前向的 logit（供外部loss使用）
        self._last_nz_logit = None

    def forward(self, x):
        # x: (B, T, in_dim)
        h, _ = self.gru(x)          # (B, T, H)
        h = h[:, -1, :]             # (B, H)
        h = self.norm(h)
        s = self.shared(h)

        mag_raw = self.head_mag(s)          # (B, 1)
        dir_raw = self.head_dir(s)          # (B, 3)
        nz_logit = self.head_nz(s)          # (B, 1)

        # 将 logit 暂存，供训练循环取用
        self._last_nz_logit = nz_logit

        mag   = F.softplus(mag_raw)              # >= 0
        direc = F.normalize(dir_raw, dim=-1)     # 单位向量
        vec   = mag * direc                      # 组装 Fx,Fy,Fz
        return vec, mag, direc



class SeqDS(Dataset):
    def __init__(self, X, Y):
        self.X = X.astype(np.float32); self.Y = Y.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])

def loss_fn(vec_pred, mag_pred, direc_pred, y_true,
            nz_th_scaled, high_th_scaled,
            weight_Fx=1.0, weight_Fy=1.8, weight_Fz=1.8,
            w_mag=1.0, w_dir=0.0, eps=1e-6):
    """
    说明：
    - 所有输入均在“线性物理域”（与门控一致）。
    - 本函数内，用 signed_log + log-cosh 来比较，稳定长尾。
    """

    # === 预计算 ===
    mag_true = torch.linalg.vector_norm(y_true, dim=-1, keepdim=True)  # (B,1) 线性域
    nz_mask  = (mag_true > nz_th_scaled).float()
    hi_mask  = (mag_true > high_th_scaled).float()

    # ---------- 1) 幅值损失：在 signed_log 域做 log-cosh ----------
    # 只对非负幅值做 signed_log 的“正半轴”版本（无需 sign）
    log_mag_pred = torch.log1p(mag_pred.clamp_min(0.0) + eps)
    log_mag_true = torch.log1p(mag_true.clamp_min(0.0) + eps)
    mag_diff = log_mag_pred - log_mag_true
    mag_loss = log_cosh(mag_diff)

    # 强力样本略加权，零样本也给一点梯度（避免“全零陷阱”）
    mag_loss = mag_loss * (1.0 + 1.0 * hi_mask)
    mag_loss = (mag_loss * (0.1 + 0.9 * nz_mask)).mean()

    # ---------- 2) 分量损失：对 Fx/Fy/Fz 用 signed_log + log-cosh ----------
    # 把预测与真值都映射到对称的 log 域
    log_vec_pred = signed_log(vec_pred, eps=eps)
    log_vec_true = signed_log(y_true,   eps=eps)
    vec_diff = log_vec_pred - log_vec_true
    comp_loss_raw = log_cosh(vec_diff)  # (B,3)

    # 分量权重
    w = torch.tensor([weight_Fx, weight_Fy, weight_Fz],
                     device=vec_pred.device, dtype=vec_pred.dtype).view(1, 3)
    comp_loss = (comp_loss_raw * w).mean()

    # ---------- 3) 方向损失（保留你原来的实现） ----------
    # 假设你已有方向的 cos-sim/余弦损失，下面只是示意：
    if w_dir > 0.0:
        # 归一化以免数值偏差
        direc_pred_n = F.normalize(direc_pred, dim=-1)
        direc_true_n = F.normalize(y_true, dim=-1).clamp_min(1e-12)  # 若你本来用 y_true 方向
        # 方向只在非零样本上起作用
        cos_sim = (direc_pred_n * direc_true_n).sum(dim=-1, keepdim=True)  # (B,1)
        dir_loss = (0.5 * (1.0 - cos_sim)) * nz_mask
        dir_loss = dir_loss.mean()
    else:
        dir_loss = torch.tensor(0.0, device=vec_pred.device)

    # ---------- 4) 稀疏惩罚 / 欠估惩罚（沿用你的写法，给个常见模板） ----------
    # (a) 稀疏：零样本不应出大力
    margin = 1.2 * nz_th_scaled
    sparse_pen = torch.relu(mag_pred - margin) * (1.0 - nz_mask)
    L_sparse = 0.8 * sparse_pen.mean()

    # (b) 欠估：在强力上更不该低估
    under_pen = torch.relu((mag_true - mag_pred)) * hi_mask
    L_under  = 0.6 * under_pen.mean()

    total = w_mag * mag_loss + comp_loss + w_dir * dir_loss + L_sparse + L_under
    return total




def linear_calibration(x_pred, y_true, nonneg=False):
    x = np.asarray(x_pred).reshape(-1,1)
    y = np.asarray(y_true).reshape(-1,1)
    X = np.concatenate([x, np.ones_like(x)], axis=1)
    lam = 1e-8
    coef = np.linalg.solve(X.T@X + lam*np.eye(2), X.T@y)
    a, b = float(coef[0,0]), float(coef[1,0])
    def apply(z):
        out = a*z + b
        if nonneg:
            out = np.maximum(out, 0.0)
        return out
    return a, b, apply


def mixed_batches(train_loader, strong_loader=None, ratio_strong=0.2, steps=None):
    """
    每个 epoch 产出固定步数的 batch（默认 = len(train_loader)）。
    会按 ratio_strong 概率从 strong_loader 抽样；若 strong 空或耗尽则回退到 train_loader。
    不让 StopIteration 逃出生成器（自动重置迭代器）。
    """
    if steps is None:
        steps = len(train_loader)  # 用训练主集的长度作为一个 epoch 的基准步数

    it_tr = iter(train_loader)
    it_str = iter(strong_loader) if strong_loader is not None else None

    for _ in range(steps):
        use_str = (it_str is not None) and (random.random() < float(ratio_strong))

        # 优先 strong，拿不到就回退到 train
        if use_str:
            try:
                xs, ys = next(it_str)
            except StopIteration:
                # strong 耗尽 -> 重置；若仍然拿不到，则回退到 train
                it_str = iter(strong_loader)
                try:
                    xs, ys = next(it_str)
                except StopIteration:
                    use_str = False  # 回退

        if not use_str:
            try:
                xs, ys = next(it_tr)
            except StopIteration:
                # train 耗尽 -> 重置后再取
                it_tr = iter(train_loader)
                xs, ys = next(it_tr)

        yield xs, ys

import math

def apply_gate(ep, nz_logit, mag, direc, nz_th_scaled, target_pos, temp=0.75, dead_k=1.0):
    """
    硬门控 + STE（基于 logits 的 Top-K 选样）：
    - 直接在 nz_logit 上取 Top-K，K = ceil(target_pos * B)
    - 前向：hard mask；反向：soft mask (温度 temp) + STE
    - deadzone 轻量，避免吃光幅值
    """
    B = nz_logit.shape[0]
    nz_logit = nz_logit.view(B, 1)

    # ---- Top-K on logits，保证每个 batch 精确选出 target_pos 比例 ----
    k = max(1, int(math.ceil(float(target_pos) * B)))
    # 取第 k 大的 logit 作为阈值
    tau_logit = torch.topk(nz_logit.flatten(), k, largest=True).values.min()

    # 硬/软 mask（STE）
    mask_hard = (nz_logit >= tau_logit).float()
    mask_soft = torch.sigmoid((nz_logit - tau_logit) / temp)
    mask = mask_hard.detach() + (mask_soft - mask_soft.detach())

    # 改成：
    if ep <= 3:
        base_mul = 0.8
    elif ep <= 8:
        base_mul = 0.7
    elif ep <= 15:
        base_mul = 0.6
    else:
        base_mul = 0.5
    deadzone = (dead_k * base_mul) * nz_th_scaled

    deadzone_vec = torch.where(
        (mask > 0.5).expand_as(mag),  # 被选中
        torch.zeros_like(mag),
        torch.full_like(mag, deadzone)
    )
    mag = torch.clamp(mag - deadzone_vec, min=0.0)
    mag = mag * mask

    # 归一化方向并合成向量
    direc = F.normalize(direc, dim=-1)
    vec = mag * direc
    return vec, mag, mask, tau_logit


# ---- utils: elementwise focal BCE (no 'reduction' arg needed) ----
def bce_focal_elementwise(logits, targets, alpha=0.5, gamma=1.5, eps=1e-8):
    if logits.dim() == 1:  logits = logits.unsqueeze(-1)
    if targets.dim() == 1: targets = targets.unsqueeze(-1)

    # 非就地 clamp（不会破坏 SigmoidBackward 的版本）
    prob = torch.sigmoid(logits).clamp(eps, 1.0 - eps)

    pt = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    loss = - alpha_t * ((1 - pt) ** gamma) * (
        targets * torch.log(prob) + (1 - targets) * torch.log(1 - prob)
    )
    return loss

# ======================
# 主流程
# ======================
def main():
    import numpy as np
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    set_seed()
    print(f"[INFO] device={device}")

    # ===================== BEGIN PATCH: 读数据→切分→阈值→强力样本→标准化→强力Loader =====================

    # 读数据
    if BALANCE_USE:
        print("[INFO] 使用硬平衡：全0 vs 非0（系统抽样）")
        df = load_and_balance_csv(DATA_DIR, time_col=TIME_COL)
    else:
        df = load_all_csv(DATA_DIR)

    # 先按“文件”切分 Train/Val（避免泄漏）
    tr_df, va_df = split_by_file(df, val_ratio_files=0.2)

    # 仅用“训练集”的物理尺度 |F| 计算阈值/缩放
    Ytr_abs = tr_df[TARGET_COLS].to_numpy(dtype=np.float64)  # 物理尺度
    all_mag_train = vec_mag(Ytr_abs)

    # 全体（仅用于统计打印，不参与阈值）
    Yall_abs = df[TARGET_COLS].to_numpy(dtype=np.float64)
    all_mag_all = vec_mag(Yall_abs)

    def compute_thresholds_from_all(all_mag,
                                    target_pos=TARGET_POS_FOR_THRESH,
                                    phys_noise_min=PHYS_NOISE_MIN,
                                    highf_fixed=HIGHF_FIXED,  # 若无固定值则为 None
                                    highf_q=HIGHF_Q):
        """
        返回：
          - nz_th_abs: 使 P(|F| > nz_th_abs) ≈ target_pos（在 all_mag 上反推）
          - p95_abs  : 在真非零(>phys_noise_min)上统计的 p95，作为缩放基准
          - high_th_abs: “强力”阈值（固定给定或全体高分位），并确保 >= 1.5 * nz_th_abs
        """
        mag = np.asarray(all_mag, dtype=np.float64)
        mag = mag[np.isfinite(mag)]
        if mag.size == 0:
            return 1e-12, 1e-12, 1e-12

        # 1) 非零阈值：按目标占比反推（在训练集分布上）
        target_pos = float(np.clip(target_pos, 1e-3, 0.2))  # 安全范围 0.1%~20%
        nz_th_abs = float(np.quantile(mag, 1.0 - target_pos))
        nz_th_abs = max(nz_th_abs, phys_noise_min)

        # 2) p95：仅在真非零(>phys_noise_min)上统计，避免伪零污染
        pos = mag[mag > phys_noise_min]
        p95_abs = float(np.quantile(pos, 0.90)) if pos.size > 0 else max(nz_th_abs, 1e-12)

        # 3) 强力阈值：固定 or 高分位（在训练集上），并拉开到至少 1.5 * nz_th_abs
        if highf_fixed is not None:
            high_th_abs = float(highf_fixed)
        else:
            high_th_abs = float(np.quantile(mag, highf_q))
        high_th_abs = float(max(high_th_abs, 1.5 * nz_th_abs))

        return nz_th_abs, p95_abs, high_th_abs

    # —— 真正计算：只用训练集的 |F| ——
    nz_th_abs, p95_abs, high_th_abs = compute_thresholds_from_all(
        all_mag_train,
        target_pos=TARGET_POS_FOR_THRESH,
        phys_noise_min=PHYS_NOISE_MIN,
        highf_fixed=HIGHF_FIXED,  # 没有固定阈值就确保 HIGHF_FIXED=None
        highf_q=HIGHF_Q
    )

    # 训练空间缩放（与后续训练/还原一致）
    y_scale = p95_abs
    nz_th_scaled = nz_th_abs / y_scale
    high_th_scaled = high_th_abs / y_scale

    print(f"[THRESH] NZ_THRESHOLD(abs)={nz_th_abs:.3e}, HIGHF_FIXED(abs)={high_th_abs:.3e}")
    print(
        f"[SCALE→TRAIN] y_scale(p95 |F|)={y_scale:.3e} -> nz_th_scaled={nz_th_scaled:.3e}, high_th_scaled={high_th_scaled:.3e}")

    # —— 统计打印（仅用于汇报，可视化；不影响阈值）——
    print(f"[STAT] 总样本数 = {len(all_mag_all)}")
    print(f"[STAT] |F| 最小 = {all_mag_all.min():.3e}, 最大 = {all_mag_all.max():.3e}")
    print(f"[STAT] |F| 均值 = {all_mag_all.mean():.3e}, 中位 = {np.median(all_mag_all):.3e}")
    nonzero_frac_all = float((all_mag_all > nz_th_abs).mean())
    print(f"[STAT] 非零样本比例(|F|>{nz_th_abs:.3e}) = {nonzero_frac_all:.3f}")

    # —— 预训练“强力”样本：只从训练集选，避免泄漏 ——
    strong_mask_tr = (all_mag_train > high_th_abs)
    strong_df = tr_df.loc[strong_mask_tr].reset_index(drop=True)

    # 仅统计用：全体占比
    strong_frac_all = float((all_mag_all > high_th_abs).mean())
    print(f"[STAT] “强力”阈值(abs) = {high_th_abs:.3e}, 全体占比 {strong_frac_all:.3f}")
    print(f"[PRE-TRAINING] 共 {len(strong_df)} 条强力样本用于预训练")

    # —— 标准化特征：用训练集均值/方差 ——
    mu = tr_df[FEATURES].mean().to_numpy()
    std = tr_df[FEATURES].std().to_numpy() + 1e-12

    def z(df_):
        X = df_[FEATURES].to_numpy()
        return (X - mu) / std

    trZ = tr_df.copy();
    trZ[FEATURES] = z(tr_df)
    vaZ = va_df.copy();
    vaZ[FEATURES] = z(va_df)

    # —— 预训练阶段（强力样本用与训练/验证一致的 z-score）——
    print("[Stage1] 预训练（强力样本，使用与训练/验证一致的 z-score）")
    strongZ = strong_df.copy()
    strongZ[FEATURES] = z(strong_df)

    seq_s = build_sequences_df(strongZ, SEQ_LEN, SEQ_STRIDE, FEATURES, TARGET_COLS)
    if seq_s is None:
        raise RuntimeError("强力样本不足构建序列")
    X_s, Y_s_abs = seq_s
    Y_s = Y_s_abs / y_scale  # 与训练主流程相同的缩放基准

    ds_s = SeqDS(X_s, Y_s)
    loader_s = DataLoader(ds_s, batch_size=BATCH_SIZE, shuffle=True)

    # —— 模型初始化 ——（如你原有）
    model = GRUReg(len(FEATURES), hidden=256, layers=1, dropout=0.1).to(device)

    # ====================== END PATCH ======================

    # ===== [PATCH-1] set_logit_bias: 按稀疏率给分类头一个先验偏置 =====
    def set_logit_bias(module, p0=0.05):
        import math
        import torch.nn as nn
        b = math.log(p0 / (1 - p0))
        for m in module.modules():
            # 如果你的分类头不是 Linear(… -> 1)，请把判断条件改成你模型里分类层的名字/类型
            if isinstance(m, nn.Linear) and m.out_features == 1:
                if m.bias is not None:
                    with torch.no_grad():
                        m.bias.fill_(b)

    # 用全局非零率（日志里 ~0.047）做先验；找不到就用0.05
    try:
        p0_prior = 0.047
    except:
        p0_prior = 0.05
    set_logit_bias(model, p0=p0_prior)

    # 冻结 GRU 层参数，仅训练 head（可选）
    for param in model.gru.parameters():
        param.requires_grad = False

    opt_s = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=LR, weight_decay=WEIGHT_DECAY)

    for ep in range(1, PRETRAIN_EPOCHS+1):
        model.train()
        tot=0; n=0
        for xb, yb in loader_s:
            xb = xb.to(device); yb = yb.to(device)
            opt_s.zero_grad(set_to_none=True)
            vec, mag, direc = model(xb)
            loss = loss_fn(vec, mag, direc, yb,
                           nz_th_scaled=0.0, high_th_scaled=0.0,
                           weight_Fx=1.0, weight_Fy=3.0, weight_Fz=3.0,
                           w_mag=1.0, w_dir=2.0)
            loss.backward()
            if CLIP_GRAD>0: nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            opt_s.step()
            tot += float(loss.item()) * xb.size(0); n += xb.size(0)
        print(f"[PreEp{ep:02d}] Loss={tot/max(1,n):.5f}")

    # 保存预训练模型
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(SAVE_DIR_BASE)/f"GRU_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir/"pretrained_model.pt")

    # ———- 微调阶段 ——
    print("[Stage2] 微调（全部样本）")
    seq_tr = build_sequences_df(trZ, SEQ_LEN, SEQ_STRIDE, FEATURES, TARGET_COLS)
    seq_va = build_sequences_df(vaZ, SEQ_LEN, SEQ_STRIDE, FEATURES, TARGET_COLS)
    assert seq_tr is not None and seq_va is not None, "序列长度不足"
    Xtr, Ytr_abs = seq_tr
    Xva, Yva_abs = seq_va
    # ==== 进入 log-domain ====
    Ytr = to_log_domain(Ytr_abs)
    Yva = to_log_domain(Yva_abs)

    nz_th_scaled = nz_th_abs   / p95_abs
    high_th_scaled = high_th_abs / p95_abs
    print(f"[THRESH] NZ_THRESHOLD(abs)={nz_th_abs:.3e}, HIGHF_FIXED(abs)={high_th_abs:.3e}")
    print(f"[SCALE→TRAIN] y_scale(p95 |F|)={p95_abs:.3e} -> nz_th_scaled={nz_th_scaled:.3e}, high_th_scaled={high_th_scaled:.3e}")

    # DataLoader（uniform sampling）
    # -------- 初始：对“序列级”做一次动态平衡（epoch=0） --------
    Xtr_bal, Ytr_bal = balance_sequences(Xtr, Ytr, nz_th_scaled, epoch=0, max_ratio=5)

    train_loader = DataLoader(SeqDS(Xtr_bal, Ytr_bal), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(SeqDS(Xva, Yva), batch_size=BATCH_SIZE, shuffle=False)

    # ---------- 新增：强力样本 DataLoader ----------
    # strong_df 是你前面用 high_th_abs 选出来的强力子集（真值尺度）
    # 用和训练集完全一致的处理方式生成序列与目标缩放
    strongZ = strong_df.copy()
    strongZ[FEATURES] = z(strong_df)  # 用训练集 mu/std 标准化
    seq_strong = build_sequences_df(strongZ, SEQ_LEN, SEQ_STRIDE, FEATURES, TARGET_COLS)
    if seq_strong is None:
        strong_loader = DataLoader(SeqDS(Xtr, Ytr), batch_size=max(8, BATCH_SIZE // 2),
                                   shuffle=True, drop_last=True)
    else:
        Xs, Ys_abs = seq_strong
        Ys = Ys_abs / p95_abs
        strong_loader = DataLoader(SeqDS(Xs, Ys), batch_size=max(8, BATCH_SIZE // 2),
                                   shuffle=True, drop_last=True)

    # 加载预训练权重
    model.load_state_dict(torch.load(out_dir/"pretrained_model.pt"))

    # 解冻部分层（如果你愿意）——这里解冻所有层
    for param in model.parameters():
        param.requires_grad = True

    base_lr = LR  # 复用你配置里的 lr
    cls_params, other_params = [], []
    for n, p in model.named_parameters():
        # 这里用关键词匹配分类头；若你的分类头命名不同，请把 'cls'/'nz' 改成你的名字关键字
        if ('cls' in n) or ('nz' in n) or ('logit' in n):
            cls_params.append(p)
        else:
            other_params.append(p)

    opt = torch.optim.AdamW([
        {'params': other_params, 'lr': base_lr},
        {'params': cls_params, 'lr': base_lr * 5.0},
    ], weight_decay=1e-4)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=base_lr * 0.1)


    best = -1e9; best_state = None; no_imp=0
    target_pos_ema = None
    alpha_ema = None
    ema_beta = 0.8

    for ep in range(1, EPOCHS + 1):

        # -------- ✅ 每个 epoch 动态重采样序列 --------
        Xtr_bal, Ytr_bal = balance_sequences(Xtr, Ytr, nz_th_scaled, epoch=ep, max_ratio=5)

        train_loader = DataLoader(
            SeqDS(Xtr_bal, Ytr_bal),
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True
        )

        # 强力样本比例（预热期更多强力）
        ratio_strong = 0.60 if ep <= 6 else (0.45 if ep <= 12 else 0.30)

        model.train()
        tot = 0.0
        n = 0

        # 以 EMA 的真实占比为基准，给一点缓冲；并设“地板”防止过保守
        if target_pos_ema is None:
            target_pos = 0.08  # 初始稍高，利于学习非零
        else:
            target_pos = float(np.clip(target_pos_ema * 1.10, 0.05, 0.12))  # ← 下限 0.05

        # ===== 是否启用门控（No-Gate 预热 1~6 epoch）=====
        use_gate_train = (ep >= 7)

        for xb, yb in mixed_batches(train_loader, strong_loader, ratio_strong=ratio_strong):
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)

            # ---------- 前向 ----------
            vec_raw, mag_raw, direc_raw = model(xb)  # 模型内部会设置 _last_nz_logit
            direc_norm = F.normalize(direc_raw, dim=-1)

            # 分类 logit（仅在启用门控/分类时使用）
            nz_logit = getattr(model, "_last_nz_logit", None)
            if nz_logit is not None and nz_logit.dim() == 1:
                nz_logit = nz_logit.unsqueeze(-1)  # (B,1)

            # ==== 温度 / deadzone 调度（门控用）====
            if ep <= 3:
                temp, dead_k = 1.5, 1.0
            elif ep <= 10:
                temp, dead_k = 1.0, 1.0
            else:
                temp, dead_k = 0.9, 0.9

            # ---------- 门控或直通 ----------
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
                # No-Gate 直通：不裁掉样本
                vec, mag = vec_raw, mag_raw
                mask = torch.ones_like(mag_raw)
                tau = torch.tensor(float('nan'))

            # ====== 计算标签与可靠掩码（分类用） ======
            y_mag_true = torch.linalg.vector_norm(yb, dim=-1, keepdim=True)  # (B,1)
            y_nz = (y_mag_true > nz_th_scaled).float()  # (B,1)
            with torch.no_grad():
                band_lo = 0.5 * nz_th_scaled
                band_hi = 1.5 * nz_th_scaled
                reliable = ((y_mag_true < band_lo) | (y_mag_true > band_hi)).float()  # (B,1)

            # ====== 分类损失（预热期直接关掉） ======
            if use_gate_train:
                # Focal α：用 EMA 稳定
                with torch.no_grad():
                    p_pos_batch = y_nz.mean().clamp(1e-4, 1 - 1e-4).item()
                    if alpha_ema is None:
                        alpha_ema = 1.0 - p_pos_batch
                    else:
                        alpha_ema = 0.9 * alpha_ema + 0.1 * (1.0 - p_pos_batch)
                alpha_pos = float(np.clip(alpha_ema, 0.7, 0.98))

                cls_element = bce_focal_elementwise(
                    nz_logit, y_nz,
                    alpha=alpha_pos, gamma=1.5
                )  # (B,1)

                reliable_sum = reliable.sum().clamp_min(1.0)
                loss_cls = (cls_element * reliable).sum() / reliable_sum

                # 分类权重调度
                if ep <= 12:
                    cls_weight = 1.2 if ep > 6 else 0.0  # 预热期 0.0；第7~12轮 1.2
                else:
                    cls_weight = 0.6
            else:
                loss_cls = torch.tensor(0.0, device=xb.device)
                cls_weight = 0.0

            # ====== 回归损失：门控强监督 + 全样本弱监督 ======
            # 方向项从 ep>=5 才加入，降低早期噪声（修正逻辑）
            w_dir_curr = 0.3 if ep >= 5 else 0.0

            # 1) 门控强监督（或直通情况下等价于全样本强监督）
            loss_reg_mask = loss_fn(
                vec, mag, direc_norm, yb,
                nz_th_scaled, high_th_scaled,
                weight_Fx=1.0, weight_Fy=1.8, weight_Fz=1.8,
                w_mag=1.0, w_dir=w_dir_curr
            )

            # 2) 全样本弱监督（不给方向项，保底梯度）
            loss_reg_all = loss_fn(
                vec_raw, mag_raw, direc_norm, yb,
                nz_th_scaled, high_th_scaled,
                weight_Fx=1.0, weight_Fy=1.8, weight_Fz=1.8,
                w_mag=1.0, w_dir=0.0
            )

            # 融合权重：预热期更偏向全样本；之后逐步回到门控
            if ep <= 6:
                w_all, w_mask = 0.7, 0.3
            else:
                t = min(1.0, max(0.0, (ep - 6) / 10.0))  # ep=6→w_all=0.2, ep=16→0.5
                w_all = 0.2 + 0.3 * t
                w_mask = 1.0 - w_all

            loss_reg = w_mask * loss_reg_mask + w_all * loss_reg_all

            # 轻量 logit 中心化正则（启用分类时才有意义）
            logit_center = (getattr(model, "_last_nz_logit") ** 2).mean() if use_gate_train else torch.tensor(0.0,
                                                                                                              device=xb.device)

            loss = loss_reg + cls_weight * loss_cls + 1e-4 * logit_center
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            tot += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        avg_loss = tot / max(1, n)
        sched.step()

        # ================= 验证 =================
        model.eval()
        with torch.no_grad():
            taus = []
            YP_list = []
            nz_true_flags = []
            nz_pred_flags_mag = []
            nz_pred_flags_mask = []
            cls_probs = []

            # 验证阶段：与训练一致，前 6 轮不用门控；之后再评估门控
            use_gate_val = (ep >= 7)

            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

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
                    tau = torch.tensor(float('nan'))

                taus.append(float(tau))

                # 预测向量（还原到绝对单位）
                YP_list.append((vec_eval.detach().cpu().numpy() * p95_abs))

                # 统计非零比例
                y_mag_true = torch.linalg.vector_norm(yb, dim=-1, keepdim=True) * p95_abs
                y_nz_flag = (y_mag_true > nz_th_abs).float().cpu()

                y_mag_pred = torch.linalg.vector_norm(vec_eval, dim=-1, keepdim=True) * p95_abs
                y_pred_flag_mag = (y_mag_pred > nz_th_abs).float().cpu()
                y_pred_flag_mask = (mask > 0.5).float().cpu()

                nz_true_flags.append(y_nz_flag)
                nz_pred_flags_mag.append(y_pred_flag_mag)
                nz_pred_flags_mask.append(y_pred_flag_mask)

                cls_probs.append(torch.sigmoid(logits).detach().cpu())

        # ==== 从 log-domain 反变换回物理尺度 ====
        YP_log = np.concatenate(YP_list, axis=0) if len(YP_list) else np.zeros((0, 3))
        YP_abs = from_log_domain(YP_log)

        nz_true = torch.cat(nz_true_flags).mean().item()
        nz_pred_mag = torch.cat(nz_pred_flags_mag).mean().item()
        nz_pred_mask = torch.cat(nz_pred_flags_mask).mean().item()
        nz_prob_mean = torch.cat(cls_probs).mean().item()
        tau_mean = np.mean(taus) if len(taus) else float('nan')

        print(
            f"[Ep{ep:03d}] cls_prob(mean)={nz_prob_mean:.3f} | "
            f"NZ(true/mask/mag)={nz_true:.3f}/{nz_pred_mask:.3f}/{nz_pred_mag:.3f} | "
            f"Tau(logit)={tau_mean:.3f}"
        )

        if target_pos_ema is None:
            target_pos_ema = nz_true  # 初始用观测值
        else:
            target_pos_ema = ema_beta * target_pos_ema + (1 - ema_beta) * nz_true

        magT  = vec_mag(Yva_abs)
        magP  = vec_mag(YP_abs)
        r2_fx = r2_score_np(Yva_abs[:,0], YP_abs[:,0])
        r2_fy = r2_score_np(Yva_abs[:,1], YP_abs[:,1])
        r2_fz = r2_score_np(Yva_abs[:,2], YP_abs[:,2])
        r2_all= r2_score_np(magT, magP)
        mask  = (magT > nz_th_abs)
        r2_nz = r2_score_np(magT[mask], magP[mask]) if mask.any() else float('nan')
        nz_true = float((magT>nz_th_abs).mean())
        nz_pred = float((magP>nz_th_abs).mean())

        print(f"[Ep{ep:03d}] Loss={avg_loss:.5f} | R2(Fx,Fy,Fz)=({r2_fx:.3f},{r2_fy:.3f},{r2_fz:.3f}) | "
              f"|F| R2_all={r2_all:.3f} R2_nz={r2_nz:.3f} | NZ(true/pred)={nz_true:.3f}/{nz_pred:.3f}")

        score = r2_nz
        if score > best + 1e-5:
            best = score; no_imp=0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= EARLYSTOP_PATIENCE:
                print(f"[EarlyStopping] stop at epoch {ep}")
                break

    # 载入最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    # 保存模型 + 画图逻辑（与原脚本类似）——你可按需补充

    torch.save({
        "state_dict": model.state_dict(),
        "mu":        mu,
        "std":       std,
        "features":  FEATURES,
        "y_scale":   p95_abs
    }, out_dir/"best_vecgru.pt")

    # 画散点（未校准）
    names = ["Fx", "Fy", "Fz"]
    for i, name in enumerate(names):
        plt.figure(figsize=(5, 5))
        plt.scatter(Yva_abs[:, i], YP_abs[:, i], s=8, alpha=0.35)
        lo = float(min(Yva_abs[:, i].min(), YP_abs[:, i].min()))
        hi = float(max(Yva_abs[:, i].max(), YP_abs[:, i].max()))
        plt.plot([lo, hi], [lo, hi], 'k--', lw=1)
        plt.xlabel(f"True {name}")
        plt.ylabel(f"Pred {name}")
        plt.title(f"VAL: {name} (R2={r2_score_np(Yva_abs[:, i], YP_abs[:, i]):.3f})")
        plt.tight_layout()
        plt.savefig(out_dir / f"pred_vs_true_{name}_VAL.png", dpi=160)
        plt.close()

    # 模长散点
    plt.figure(figsize=(5, 5))
    lo = float(min(magT.min(), magP.min()))
    hi = float(max(magT.max(), magP.max()))
    plt.scatter(magT, magP, s=8, alpha=0.35)
    plt.plot([lo, hi], [lo, hi], 'k--', lw=1)
    plt.xlabel("|F| True")
    plt.ylabel("|F| Pred")
    plt.title(f"VAL: |F| (R2_all={r2_all:.3f}, R2_nz={r2_nz:.3f})")
    plt.tight_layout()
    plt.savefig(out_dir / "pred_vs_true_Fmag_VAL.png", dpi=160)
    plt.close()

    # 残差直方图
    plt.figure(figsize=(7, 4))
    plt.hist(magP - magT, bins=80, alpha=0.9)
    plt.xlabel("Residual (|F|_pred - |F|_true)")
    plt.ylabel("Count")
    plt.title("VAL: |F| Residual Histogram")
    plt.tight_layout()
    plt.savefig(out_dir / "residual_hist_Fmag_VAL.png", dpi=160)
    plt.close()

    # 线性标定 |F|
    a_m, b_m, fn_mag = linear_calibration(magP, magT, nonneg=True)
    magP_cal = fn_mag(magP)
    r2_all_c = r2_score_np(magT, magP_cal)
    mask = (magT > nz_th_abs)
    r2_nz_c = r2_score_np(magT[mask], magP_cal[mask]) if mask.any() else float('nan')

    plt.figure(figsize=(5, 5))
    plt.scatter(magT, magP_cal, s=8, alpha=0.35)
    plt.plot([lo, hi], [lo, hi], 'k--', lw=1)
    plt.xlabel("|F| True")
    plt.ylabel("|F| Pred (Calib)")
    plt.title(f"VAL: |F| Calibrated (R2_all={r2_all_c:.3f}, R2_nz={r2_nz_c:.3f})")
    plt.tight_layout()
    plt.savefig(out_dir / "pred_vs_true_Fmag_VAL_CAL.png", dpi=160)
    plt.close()

    # 分量标定（Fy,Fz）
    for i, (name, do_cal) in enumerate([("Fx", False), ("Fy", True), ("Fz", True)]):
        y_true = Yva_abs[:, i]
        y_pred = YP_abs[:, i]
        if do_cal:
            a, b, fn = linear_calibration(y_pred, y_true, nonneg=False)
            y_cal = fn(y_pred)
            r2c = r2_score_np(y_true, y_cal)
            plt.figure(figsize=(5, 5))
            lo = float(min(y_true.min(), y_cal.min()))
            hi = float(max(y_true.max(), y_cal.max()))
            plt.scatter(y_true, y_cal, s=8, alpha=0.35)
            plt.plot([lo, hi], [lo, hi], 'k--', lw=1)
            plt.xlabel(f"True {name}")
            plt.ylabel(f"Pred {name} (Calib)")
            plt.title(f"VAL: {name} (Calib R2={r2c:.3f})")
            plt.tight_layout()
            plt.savefig(out_dir / f"pred_vs_true_{name}_VAL_CAL.png", dpi=160)
            plt.close()
        else:
            plt.figure(figsize=(5, 5))
            lo = float(min(y_true.min(), y_pred.min()))
            hi = float(max(y_true.max(), y_pred.max()))
            plt.scatter(y_true, y_pred, s=8, alpha=0.35)
            plt.plot([lo, hi], [lo, hi], 'k--', lw=1)
            plt.xlabel(f"True {name}")
            plt.ylabel(f"Pred {name}")
            plt.title(f"VAL: {name} (R2={r2_score_np(y_true, y_pred):.3f})")
            plt.tight_layout()
            plt.savefig(out_dir / f"pred_vs_true_{name}_VAL_CAL.png", dpi=160)
            plt.close()


    print(f"[CALIB] |F|_calib = max(0, {a_m:.6f} * |F|_pred + {b_m:.6f})")

    # 外置解释模块
    from interpert_addons import run_gam, run_pls, run_linear_ridge
    run_gam(trZ, tr_df, FEATURES, TARGET_COLS, out_dir, gam_splines=10, gam_lam=0.3, gam_max_samples=200_000)
    run_pls(trZ, tr_df, FEATURES, TARGET_COLS, out_dir, n_comp=None, topk=15)
    run_linear_ridge(trZ, tr_df, FEATURES, TARGET_COLS, out_dir, alphas=None, topk=15)
    # 分量 MAE / RMSE
    mae_fx = mean_absolute_error(Yva_abs[:, 0], YP_abs[:, 0])
    mae_fy = mean_absolute_error(Yva_abs[:, 1], YP_abs[:, 1])
    mae_fz = mean_absolute_error(Yva_abs[:, 2], YP_abs[:, 2])

    mse_fx = mean_squared_error(Yva_abs[:, 0], YP_abs[:, 0])
    mse_fy = mean_squared_error(Yva_abs[:, 1], YP_abs[:, 1])
    mse_fz = mean_squared_error(Yva_abs[:, 2], YP_abs[:, 2])

    rmse_fx = np.sqrt(mse_fx)
    rmse_fy = np.sqrt(mse_fy)
    rmse_fz = np.sqrt(mse_fz)

    # 模长 MAE / RMSE
    mae_mag = mean_absolute_error(magT, magP)
    mse_mag = mean_squared_error(magT, magP)
    rmse_mag = np.sqrt(mse_mag)

    print(f"MAE_Fx={mae_fx:.6e}, MAE_Fy={mae_fy:.6e}, MAE_Fz={mae_fz:.6e}")
    print(f"RMSE_Fx={rmse_fx:.6e}, RMSE_Fy={rmse_fy:.6e}, RMSE_Fz={rmse_fz:.6e}")
    print(f"MAE_|F|={mae_mag:.6e}, RMSE_|F|={rmse_mag:.6e}")

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    # 分量 MAE / RMSE
    mae_fx = mean_absolute_error(Yva_abs[:, 0], YP_abs[:, 0])
    mae_fy = mean_absolute_error(Yva_abs[:, 1], YP_abs[:, 1])
    mae_fz = mean_absolute_error(Yva_abs[:, 2], YP_abs[:, 2])

    mse_fx = mean_squared_error(Yva_abs[:, 0], YP_abs[:, 0])
    mse_fy = mean_squared_error(Yva_abs[:, 1], YP_abs[:, 1])
    mse_fz = mean_squared_error(Yva_abs[:, 2], YP_abs[:, 2])

    rmse_fx = np.sqrt(mse_fx)
    rmse_fy = np.sqrt(mse_fy)
    rmse_fz = np.sqrt(mse_fz)

    # 模长 MAE / RMSE
    mae_mag = mean_absolute_error(magT, magP)
    mse_mag = mean_squared_error(magT, magP)
    rmse_mag = np.sqrt(mse_mag)

    print(f"MAE_Fx={mae_fx:.6e}, MAE_Fy={mae_fy:.6e}, MAE_Fz={mae_fz:.6e}")
    print(f"RMSE_Fx={rmse_fx:.6e}, RMSE_Fy={rmse_fy:.6e}, RMSE_Fz={rmse_fz:.6e}")
    print(f"MAE_|F|={mae_mag:.6e}, RMSE_|F|={rmse_mag:.6e}")

    r_fx, _ = pearsonr(Yva_abs[:, 0], YP_abs[:, 0])
    r_fy, _ = pearsonr(Yva_abs[:, 1], YP_abs[:, 1])
    r_fz, _ = pearsonr(Yva_abs[:, 2], YP_abs[:, 2])
    r_mag, _ = pearsonr(magT, magP)

    print(f"Pearson r Fx = {r_fx:.3f}, r Fy = {r_fy:.3f}, r Fz = {r_fz:.3f}, r |F| = {r_mag:.3f}")

    # 写入 metrics CSV（假设你后面那段有保存指标）
    pd.DataFrame([{
        "R2_Fx": r2_fx,
        "R2_Fy": r2_fy,
        "R2_Fz": r2_fz,
        "Pearson_r_Fx": r_fx,
        "Pearson_r_Fy": r_fy,
        "Pearson_r_Fz": r_fz,
        "Pearson_r_|F|": r_mag,
        "MAE_Fx": mae_fx,
        "MAE_Fy": mae_fy,
        "MAE_Fz": mae_fz,
        "RMSE_Fx": rmse_fx,
        "RMSE_Fy": rmse_fy,
        "RMSE_Fz": rmse_fz,
        "R2_|F|_all_raw": r2_all,
        "R2_|F|_nz_raw": r2_nz,
        "MAE_|F|": mae_mag,
        "RMSE_|F|": rmse_mag,
        "R2_|F|_all_cal": r2_all_c,
        "R2_|F|_nz_cal": r2_nz_c,
        "NZ_true_ratio": float((magT > nz_th_abs).mean()),
        "NZ_pred_ratio": float((magP > nz_th_abs).mean()),
        "calib_|F|_a": a_m,
        "calib_|F|_b": b_m,
        "y_scale_p95": p95_abs,  # 或者 p95_abs
        "highF_abs_value": high_th_abs
    }]).to_csv(out_dir / "val_metrics.csv", index=False)

    print("[DONE]")

    # 取 d 列全量（可能包含比 Yva_abs 更多行）
    d_full = va_df["d"].to_numpy()
    N_full = len(d_full)
    N_y = Yva_abs.shape[0]
    print(f"[DEBUG] 全 va_df d 长度 = {N_full}, Yva_abs 长度 = {N_y}")

    # 对齐：假定 build_sequences_df 丢弃前 SEQ_LEN-1 行
    offset = SEQ_LEN - 1
    if N_full != N_y:
        print(f"[DEBUG] 使用 offset={offset} 来对齐 d 数组")
        d_va_arr = d_full[offset: offset + N_y]
        if len(d_va_arr) != N_y:
            raise ValueError(f"对齐失败：截取后的 d_va_arr 长度 = {len(d_va_arr)}, 不是 {N_y}")
    else:
        d_va_arr = d_full

    assert d_va_arr.shape[0] == N_y

    # 计算模长
    magT = np.sqrt((Yva_abs ** 2).sum(axis=1))
    magP = np.sqrt((YP_abs ** 2).sum(axis=1))

    # 分箱设置
    num_bins = 10
    bins = np.linspace(d_va_arr.min(), d_va_arr.max(), num_bins + 1)
    bin_indices = np.digitize(d_va_arr, bins)

    results = []
    for b in range(1, num_bins + 1):
        mask = (bin_indices == b)
        count_b = mask.sum()
        if count_b < 20:
            continue

        seg_true = Yva_abs[mask]
        seg_pred = YP_abs[mask]
        seg_magT = magT[mask]
        seg_magP = magP[mask]
        seg_d = d_va_arr[mask]

        mae_mag = mean_absolute_error(seg_magT, seg_magP)
        rmse_mag = np.sqrt(mean_squared_error(seg_magT, seg_magP))
        r2_mag = r2_score(seg_magT, seg_magP)

        mae_fx = mean_absolute_error(seg_true[:, 0], seg_pred[:, 0])
        mae_fy = mean_absolute_error(seg_true[:, 1], seg_pred[:, 1])
        mae_fz = mean_absolute_error(seg_true[:, 2], seg_pred[:, 2])

        # Pearson 相关系数：|F| 与 d
        try:
            r_mag_d, p_mag_d = pearsonr(seg_magT, seg_d)
        except Exception as e:
            r_mag_d, p_mag_d = np.nan, np.nan

        print(f"|F|与d Pearson 相关系数:{r_mag_d}, {p_mag_d}")

        results.append({
            "bin": b,
            "d_min": bins[b - 1],
            "d_max": bins[b],
            "count": count_b,
            "MAE_|F|": mae_mag,
            "RMSE_|F|": rmse_mag,
            "R2_|F|": r2_mag,
            "MAE_Fx": mae_fx,
            "MAE_Fy": mae_fy,
            "MAE_Fz": mae_fz,
            "Pearson_r_|F|_vs_d": r_mag_d,
            "Pearson_pval_|F|_vs_d": p_mag_d,
        })

    df_bins = pd.DataFrame(results)
    out_csv = Path(SAVE_DIR_BASE) / f"GRU_{run_id}" / "bin_error_vs_distance_d.csv"
    df_bins.to_csv(out_csv, index=False)
    print(f"[SAVE] 分箱误差结果已保存 → {out_csv}")

    # 绘图：MAE & RMSE 随距离变化
    plt.figure(figsize=(6, 4))
    mid_points = (df_bins["d_min"] + df_bins["d_max"]) / 2
    plt.plot(mid_points, df_bins["MAE_|F|"], marker='o', label="MAE(|F|)")
    plt.plot(mid_points, df_bins["RMSE_|F|"], marker='s', linestyle='--', label="RMSE(|F|)")
    plt.xlabel("Distance d")
    plt.ylabel("Error (|F|)")
    plt.title("Error vs Distance d")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_png = Path(SAVE_DIR_BASE) / f"GRU_{run_id}" / "error_vs_distance_d.png"
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[SAVE] 图像已保存 → {out_png}")

    # 假设 va_df 包含四元数列：pair_0_1_qrel_x, _y, _z, _w
    q_cols = ["pair_0_1_qrel_x", "pair_0_1_qrel_y", "pair_0_1_qrel_z", "pair_0_1_qrel_w"]
    q_arr = va_df[q_cols].to_numpy(dtype=float)
    # 计算角差 θ（单位：弧度）: θ = 2 * arccos(|w|)
    w = q_arr[:, 3]
    theta = 2.0 * np.arccos(np.clip(np.abs(w), 0.0, 1.0))
    # 转为度数（可选）
    theta_deg = np.degrees(theta)

    # 确保角度数组长度与 Yva_abs 对应（如同距离 d 做过的对齐逻辑）
    N_y = Yva_abs.shape[0]
    N_q = theta_deg.shape[0]
    print(f"[DEBUG] 全 va_df 角差长度 = {N_q}, Yva_abs 长度 = {N_y}")
    if N_q != N_y:
        offset = SEQ_LEN - 1
        print(f"[DEBUG] 角度差数组长度不一致，使用 offset={offset} 对齐")
        theta_arr = theta_deg[offset: offset + N_y]
        if theta_arr.shape[0] != N_y:
            raise ValueError(f"对齐失败：theta_arr 长度 = {theta_arr.shape[0]}, 不是 {N_y}")
    else:
        theta_arr = theta_deg

    assert theta_arr.shape[0] == N_y

    # 误差 – 使用 |F| 模长误差为主，也可以做 Fx/Fy/Fz
    magT = np.sqrt((Yva_abs ** 2).sum(axis=1))
    magP = np.sqrt((YP_abs ** 2).sum(axis=1))
    err_mag = np.abs(magP - magT)

    # 角度分箱
    num_bins_phi = 10
    bins_phi = np.linspace(theta_arr.min(), theta_arr.max(), num_bins_phi + 1)
    bin_idx_phi = np.digitize(theta_arr, bins_phi)

    results_phi = []
    for b in range(1, num_bins_phi + 1):
        mask = (bin_idx_phi == b)
        count_b = mask.sum()
        if count_b < 20:
            # 样本太少时跳过
            continue

        seg_magT = magT[mask]
        seg_magP = magP[mask]
        seg_true = Yva_abs[mask]
        seg_pred = YP_abs[mask]
        seg_phi = theta_arr[mask]

        mae_mag = mean_absolute_error(seg_magT, seg_magP)
        rmse_mag = np.sqrt(mean_squared_error(seg_magT, seg_magP))
        r2_mag = r2_score(seg_magT, seg_magP)

        mae_fx = mean_absolute_error(seg_true[:, 0], seg_pred[:, 0])
        mae_fy = mean_absolute_error(seg_true[:, 1], seg_pred[:, 1])
        mae_fz = mean_absolute_error(seg_true[:, 2], seg_pred[:, 2])

        # Pearson 相关系数：|F| 与 ϕ
        try:
            r_mag_phi, p_mag_phi = pearsonr(seg_magT, seg_phi)
        except Exception as e:
            r_mag_phi, p_mag_phi = np.nan, np.nan

        print(f"|F| 与 ϕ Pearson 相关系数:{r_mag_phi}, {p_mag_phi}")

        results_phi.append({
            "bin_phi": b,
            "phi_lo_deg": bins_phi[b - 1],
            "phi_hi_deg": bins_phi[b],
            "count": count_b,
            "MAE_|F|": mae_mag,
            "RMSE_|F|": rmse_mag,
            "R2_|F|": r2_mag,
            "MAE_Fx": mae_fx,
            "MAE_Fy": mae_fy,
            "MAE_Fz": mae_fz,
            "Pearson_r_|F|_vs_phi": r_mag_phi,
            "Pearson_pval_|F|_vs_phi": p_mag_phi,
        })

    df_bins_phi = pd.DataFrame(results_phi)
    out_csv_phi = Path(SAVE_DIR_BASE) / f"GRU_{run_id}" / "bin_error_vs_angle_phi.csv"
    df_bins_phi.to_csv(out_csv_phi, index=False)
    print(f"[SAVE] 角度分箱误差结果已保存 → {out_csv_phi}")

    # 绘图：MAE & RMSE 随 ϕ 变化
    plt.figure(figsize=(6, 4))
    mid_points_phi = (df_bins_phi["phi_lo_deg"] + df_bins_phi["phi_hi_deg"]) / 2
    plt.plot(mid_points_phi, df_bins_phi["MAE_|F|"], marker='o', label="MAE(|F|)")
    plt.plot(mid_points_phi, df_bins_phi["RMSE_|F|"], marker='s', linestyle='--', label="RMSE(|F|)")
    plt.xlabel("Orientation angle difference φ (deg)")
    plt.ylabel("Error (|F|)")
    plt.title("Error vs Orientation φ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_png_phi = Path(SAVE_DIR_BASE) / f"GRU_{run_id}" / "error_vs_angle_phi.png"
    plt.savefig(out_png_phi, dpi=160)
    plt.close()
    print(f"[SAVE] 图像已保存 → {out_png_phi}")

if __name__ == "__main__":
    main()
