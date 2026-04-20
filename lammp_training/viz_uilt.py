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
DATA_DIR   = "nn_input_data_xy"   # 你的 CSV 目录
GLOB_PAT   = "*.csv"

FEATURES = [
    "d","r0","h0",
    "pair_0_1_qrel_x","pair_0_1_qrel_y","pair_0_1_qrel_z","pair_0_1_qrel_w"
]
TARGET_COLS = ["U_int_Fx","U_int_Fy","U_int_Fz"]

# 序列
SEQ_LEN    = 5
SEQ_STRIDE = 1

# 训练
BATCH_SIZE       = 1024
EPOCHS           = 200
PRETRAIN_EPOCHS  = 10   # 预训练阶段 epoch 数
LR               = 3e-4
WEIGHT_DECAY     = 1e-4
CLIP_GRAD        = 1.0
SEED             = 2025
device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# —— 目标缩放：训练时按 p95(|F|) 缩放到 ~1，再在评估/画图时反缩放回来 ——
Y_P95_SCALE = None  # 设为 None 表示自动用训练集 p95

# —— 非零判定阈值（物理单位）——
NZ_THRESHOLD_ABS = 1e-4

# —— 固定“高|F|”阈值（物理单位）——
HIGHF_FIXED = 1e-3

# —— 训练时的抽样目标（暂关闭加权采样机制）——
#ZERO_TARGET_FRAC = 0.30  # 暂时不使用

# —— 损失：两种 loss 同时启用 + 欠估惩罚——
BETA_SMOOTHL1        = 0.1
USE_DIR_LOSS         = True
DIR_LOSS_WEIGHT      = 0.5

USE_UNDER_PENALTY    = True
UNDER_PENALTY_WEIGHT = 1.0

# —— 损失内样本加权（暂关闭）——
USE_LOSS_WEIGHT = False
W_NONZERO       = 2.0
W_HIGHF         = 4.0

# 早停
EARLYSTOP_PATIENCE = 20

# 输出
SAVE_DIR_BASE = "./artifacts_vec"

BALANCE_USE        = False
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

def balance_by_target_mag(df: pd.DataFrame, target_cols: list,
                          low_quantile=0.50, high_quantile=0.90,
                          undersample_frac=0.5, oversample_factor=3,
                          noise_level=1e-4, random_state=42) -> pd.DataFrame:
    """
    按目标模长(|F|)做欠采样 + 重采样。
      df: 原训练DataFrame，必须包含 target_cols
      target_cols: 列名列表，如 ["Fx","Fy","Fz"]
    返回：重采样后的新的DataFrame
    """
    # 计算模长
    mags = np.sqrt((df[target_cols].to_numpy(dtype=float) ** 2).sum(axis=1))
    df = df.copy()
    df["_mag"] = mags

    # 定义分区
    q_low  = np.quantile(mags, low_quantile)
    q_high = np.quantile(mags, high_quantile)
    mask_low  = mags <= q_low
    mask_mid  = (mags > q_low) & (mags < q_high)
    mask_high = mags >= q_high

    df_low  = df[mask_low]
    df_mid  = df[mask_mid]
    df_high = df[mask_high]

    print(f"[BALANCE] low-mag count={len(df_low)}, mid={len(df_mid)}, high={len(df_high)}")

    # 欠采样多数区 (low 区)
    df_low_u = df_low.sample(frac=undersample_frac, random_state=random_state)

    # 重采样少样本区 (high 区) ——简单复制 +加噪声
    df_high_o = pd.concat([df_high] * oversample_factor, ignore_index=True)
    # 加少量噪声（只扰动特征，可选也扰动目标）
    feature_cols = [c for c in df.columns if c not in (target_cols + ["__file__", "_mag"])]
    for c in feature_cols:
        df_high_o[c] += np.random.normal(loc=0.0, scale=noise_level, size=len(df_high_o))

    # 合并
    df_bal = pd.concat([df_low_u, df_mid, df_high_o], ignore_index=True)
    df_bal = df_bal.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print(f"[BALANCE] after balance total count={len(df_bal)}")

    # 删除辅助列
    df_bal = df_bal.drop(columns=["_mag"])
    return df_bal

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

# ======================
# 模型 & 损失
# ======================
class ResidualBlock(nn.Module):
    """一个简单的残差块：输入维度 = 输出维度，h → h + F(h)"""
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        out = self.norm(out + identity)
        return out

class GRURegResidual(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=1, dropout=0.1, num_res_blocks=2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True,
                          dropout=(dropout if layers > 1 else 0.0))
        self.norm0 = nn.LayerNorm(hidden)
        # 增加多个残差块
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden, hidden_dim=hidden, dropout=dropout) for _ in range(num_res_blocks)]
        )
        # 输出头（分量预测）
        self.head_Fx = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)
        )
        self.head_Fy = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)
        )
        self.head_Fz = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x):
        out, _ = self.gru(x)
        h = out[:, -1, :]             # 取最后一个 time-step
        h = self.norm0(h)
        h = self.res_blocks(h)        # 残差块处理
        fx = self.head_Fx(h).squeeze(-1)
        fy = self.head_Fy(h).squeeze(-1)
        fz = self.head_Fz(h).squeeze(-1)
        y_pred = torch.stack([fx, fy, fz], dim=-1)
        return y_pred, fx, fy, fz


class SeqDS(Dataset):
    def __init__(self, X, Y):
        self.X = X.astype(np.float32); self.Y = Y.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])

def compute_loss(y_pred_vec, y_true_vec,
                 nz_th_scaled, high_th_scaled,
                 weight_Fx=1.0, weight_Fy=2.0, weight_Fz=2.0):
    fx_pred = y_pred_vec[:,0]
    fy_pred = y_pred_vec[:,1]
    fz_pred = y_pred_vec[:,2]
    fx_true = y_true_vec[:,0]
    fy_true = y_true_vec[:,1]
    fz_true = y_true_vec[:,2]

    loss_fx = F.smooth_l1_loss(fx_pred, fx_true, beta=BETA_SMOOTHL1, reduction='mean')
    loss_fy = F.smooth_l1_loss(fy_pred, fy_true, beta=BETA_SMOOTHL1, reduction='mean')
    loss_fz = F.smooth_l1_loss(fz_pred, fz_true, beta=BETA_SMOOTHL1, reduction='mean')
    loss_comp = weight_Fx*loss_fx + weight_Fy*loss_fy + weight_Fz*loss_fz

    t_mag = torch.sqrt(torch.clamp((y_true_vec**2).sum(dim=-1), min=1e-18))
    p_mag = torch.sqrt(torch.clamp((y_pred_vec**2).sum(dim=-1), min=1e-18))

    mag_loss = F.smooth_l1_loss(torch.log1p(p_mag), torch.log1p(t_mag), reduction='mean')

    if USE_DIR_LOSS:
        dot = (y_pred_vec * y_true_vec).sum(dim=-1)
        cos_sim = dot / (p_mag * t_mag + 1e-12)
        dir_loss = (1.0 - cos_sim).clamp(min=0.0)
        mask_dir = (t_mag > nz_th_scaled).float()
        dir_loss = (dir_loss * mask_dir).mean()
    else:
        dir_loss = torch.tensor(0.0, device=y_pred_vec.device)

    total_loss = loss_comp + mag_loss + DIR_LOSS_WEIGHT * dir_loss

    if USE_UNDER_PENALTY:
        under = torch.clamp(t_mag - p_mag, min=0.0)
        high  = (t_mag >= high_th_scaled).float()
        under_pen = (under * high).mean()
        total_loss = total_loss + UNDER_PENALTY_WEIGHT * under_pen

    return total_loss

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

# ======================
# 主流程
# ======================
def main():
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    set_seed()
    print(f"[INFO] device={device}")

    # 读数据
    if BALANCE_USE:
        print("[INFO] 使用硬平衡：全0 vs 非0（系统抽样）")
        df = load_and_balance_csv(DATA_DIR, time_col=TIME_COL)
    else:
        df = load_all_csv(DATA_DIR)

    # 统计目标分布
    all_Y_abs = df[TARGET_COLS].to_numpy(dtype=np.float64)
    all_mag   = vec_mag(all_Y_abs)
    print(f"[STAT] 总样本数 = {len(all_mag)}")
    print(f"[STAT] |F| 最小 = {all_mag.min():.3e}, 最大 = {all_mag.max():.3e}")
    print(f"[STAT] |F| 均值 = {all_mag.mean():.3e}, 中位 = {np.median(all_mag):.3e}")
    nonzero_frac = float((all_mag > NZ_THRESHOLD_ABS).mean())
    print(f"[STAT] 非零样本比例(|F|>{NZ_THRESHOLD_ABS:.3e}) = {nonzero_frac:.3f}")
    strong_thresh = float(np.quantile(all_mag, 0.95))
    strong_frac  = float((all_mag >= strong_thresh).mean())
    print(f"[STAT] “强力”阈值(abs) = {strong_thresh:.3e}, 占比 {strong_frac:.3f}")

    # —— 先切分训练／验证 ——
    tr_df, va_df = split_by_file(df, val_ratio_files=0.2)

    # —— 在欠采样 + 重采样逻辑前，打印初始各区样本数 ——
    Ytr_abs = tr_df[TARGET_COLS].to_numpy(dtype=np.float64)
    mag_tr = vec_mag(Ytr_abs)
    low_q = np.quantile(mag_tr, 0.70)
    high_q = np.quantile(mag_tr, 0.95)

    mask_low = mag_tr <= low_q
    mask_mid = (mag_tr > low_q) & (mag_tr < high_q)
    mask_high = mag_tr >= high_q

    df_low = tr_df.loc[mask_low].copy()
    df_mid = tr_df.loc[mask_mid].copy()
    df_high = tr_df.loc[mask_high].copy()

    print(
        f"[DEBUG] 原始 train 中各区：low={len(df_low)}, mid={len(df_mid)}, high={len(df_high)}, 合计={len(df_low) + len(df_mid) + len(df_high)}")

    # 欠采样 low 区
    undersample_frac = 0.1
    df_low_u = df_low.sample(frac=undersample_frac, random_state=SEED)
    print(f"[DEBUG] 欠采样后 low 区数 = {len(df_low_u)}")

    # 重采样 high 区
    oversample_factor = 2
    df_high_o = pd.concat([df_high] * oversample_factor, ignore_index=True)
    print(f"[DEBUG] 重采样后 high 区数 = {len(df_high_o)} (原 high={len(df_high)}, factor={oversample_factor})")

    # 合并前打印中区
    print(f"[DEBUG] mid 区数 = {len(df_mid)}")

    # 合并新的训练集
    tr_df_bal = pd.concat([df_low_u, df_mid, df_high_o], ignore_index=True)
    print(f"[DEBUG] 合并后 tr_df_bal 数 = {len(tr_df_bal)}")

    # 如需要还可以打印比例
    total_pre = len(tr_df)
    total_bal = len(tr_df_bal)
    print(f"[DEBUG] 原始 train 总 {total_pre} -> 平衡后总 {total_bal}, 倍数 = {total_bal / total_pre:.2f}")

    print(f"[BALANCE] train low count={len(df_low)}, mid={len(df_mid)}, high={len(df_high)}")

    undersample_frac   = 0.1
    df_low_u = df_low.sample(frac=undersample_frac, random_state=SEED)

    oversample_factor  = 2
    df_high_o = pd.concat([df_high] * oversample_factor, ignore_index=True)
    noise_level        = 1e-4
    feature_cols       = [c for c in df_high_o.columns if c in FEATURES]
    for c in feature_cols:
        df_high_o[c] += np.random.normal(loc=0.0, scale=noise_level, size=len(df_high_o))

    tr_df_bal = pd.concat([df_low_u, df_mid, df_high_o], ignore_index=True)
    tr_df_bal = tr_df_bal.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"[BALANCE] balance后的 train 总样本数 = {len(tr_df_bal)}")

    # —— 特征标准化（用平衡后训练集）——
    p95_abs = float(np.quantile(all_mag, 0.95))
    print(f"[INFO] p95(|F|) = {p95_abs:.3e}")

    mu  = tr_df_bal[FEATURES].mean().to_numpy()
    std = tr_df_bal[FEATURES].std().to_numpy() + 1e-12

    def z(df_):
        X = df_[FEATURES].to_numpy()
        return (X - mu) / std

    trZ = tr_df_bal.copy()
    trZ[FEATURES] = z(tr_df_bal)
    vaZ = va_df.copy()
    vaZ[FEATURES] = z(va_df)

    # 构建序列
    tr_seq = build_sequences_df(trZ, SEQ_LEN, SEQ_STRIDE, FEATURES, TARGET_COLS)
    va_seq = build_sequences_df(vaZ, SEQ_LEN, SEQ_STRIDE, FEATURES, TARGET_COLS)
    assert tr_seq is not None and va_seq is not None, "序列长度不足"

    Xtr, Ytr_abs = tr_seq
    Xva, Yva_abs = va_seq

    # 缩放目标
    y_scale = max(p95_abs, 1e-8)
    Ytr    = Ytr_abs / y_scale
    Yva    = Yva_abs / y_scale

    # —— 预训练阶段 （强力子集）——
    strong_df = df.loc[ all_mag >= strong_thresh ].reset_index(drop=True)
    print(f"[PRE-TRAINING] 共 {len(strong_df)} 条强力样本用于预训练")

    seq_s     = build_sequences_df(strong_df, SEQ_LEN, SEQ_STRIDE, FEATURES, TARGET_COLS)
    if seq_s is None:
        raise RuntimeError("强力样本不足构建序列")
    X_s, Y_s_abs = seq_s
    Y_s = Y_s_abs / p95_abs
    ds_s  = SeqDS(X_s, Y_s)
    loader_s = DataLoader(ds_s, batch_size=BATCH_SIZE, shuffle=True)

    model = GRURegResidual(len(FEATURES), hidden=128, layers=1, dropout=0.1).to(device)

    # 可选：冻结 GRU 层，仅训练 head
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
            y_pred_vec, _, _, _ = model(xb)
            loss = compute_loss(y_pred_vec, yb,
                                nz_th_scaled=0.0, high_th_scaled=0.0,
                                weight_Fx=1.0, weight_Fy=1.0, weight_Fz=1.0)
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

    # —— 微调阶段（全部样本）——
    print("[Stage2] 微调（全部样本）")
    nz_th_abs = NZ_THRESHOLD_ABS
    high_th_abs = HIGHF_FIXED
    nz_th_scaled = nz_th_abs / p95_abs
    high_th_scaled = high_th_abs / p95_abs
    print(f"[THRESH] NZ_THRESHOLD(abs)={nz_th_abs:.3e}, HIGHF_FIXED(abs)={high_th_abs:.3e}")
    print(
        f"[SCALE ] y_scale(p95 |F|)={p95_abs:.3e} -> nz_th_scaled={nz_th_scaled:.3f}, high_th_scaled={high_th_scaled:.3f}")

    train_loader = DataLoader(SeqDS(Xtr, Ytr), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(SeqDS(Xva, Yva), batch_size=BATCH_SIZE, shuffle=False)

    model.load_state_dict(torch.load(out_dir/"pretrained_model.pt"))

    for param in model.parameters():
        param.requires_grad = True

    opt = torch.optim.AdamW(model.parameters(), lr=LR*0.5, weight_decay=WEIGHT_DECAY)

    best=-1e9; best_state=None; no_imp=0
    for ep in range(1, EPOCHS+1):
        model.train()
        tot=0; n=0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            y_pred_vec, _, _, _ = model(xb)
            loss = compute_loss(y_pred_vec, yb,
                                nz_th_scaled, high_th_scaled,
                                weight_Fx=1.0, weight_Fy=2.0, weight_Fz=2.0)
            loss.backward()
            if CLIP_GRAD>0: nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            opt.step()
            tot += float(loss.item()) * xb.size(0); n += xb.size(0)
        avg_loss = tot / max(1,n)

        model.eval()
        YP_list = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                y_pred_vec, _, _, _ = model(xb)
                YP_list.append(y_pred_vec.cpu().numpy() * y_scale)
        YP_abs = np.concatenate(YP_list, axis=0)

        magT  = vec_mag(Yva_abs)
        magP  = vec_mag(YP_abs)
        r2_fx = r2_score_np(Yva_abs[:,0], YP_abs[:,0])
        r2_fy = r2_score_np(Yva_abs[:,1], YP_abs[:,1])
        r2_fz = r2_score_np(Yva_abs[:,2], YP_abs[:,2])
        r2_all= r2_score_np(magT, magP)
        mask  = (magT > NZ_THRESHOLD_ABS)
        r2_nz = r2_score_np(magT[mask], magP[mask]) if mask.any() else float('nan')
        nz_true = float((magT>NZ_THRESHOLD_ABS).mean())
        nz_pred = float((magP>NZ_THRESHOLD_ABS).mean())

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
    mask = (magT > NZ_THRESHOLD_ABS)
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
    run_gam(trZ, tr_df_bal, FEATURES, TARGET_COLS, out_dir, gam_splines=10, gam_lam=0.3, gam_max_samples=200_000)
    run_pls(trZ, tr_df_bal, FEATURES, TARGET_COLS, out_dir, n_comp=None, topk=15)
    run_linear_ridge(trZ, tr_df_bal, FEATURES, TARGET_COLS, out_dir, alphas=None, topk=15)
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
        "NZ_true_ratio": float((magT > NZ_THRESHOLD_ABS).mean()),
        "NZ_pred_ratio": float((magP > NZ_THRESHOLD_ABS).mean()),
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
