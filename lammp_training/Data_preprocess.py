# -*- coding: utf-8 -*-
# ---------- OpenMP/MKL on Windows: 避免重复加载崩溃 ----------
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import glob, time, random, warnings
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
DATA_DIR   = "nn_input_data_xy"  # 你的 CSV 目录
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
BATCH_SIZE = 1024
EPOCHS     = 60
LR         = 3e-4
WEIGHT_DECAY = 1e-4
CLIP_GRAD  = 1.0
SEED       = 2025
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# —— 目标缩放：训练时按 p95(|F|) 缩放到 ~1，再在评估/画图时反缩放回来 ——
Y_P95_SCALE = None  # 设为 None 表示自动用“训练集” p95（已修复泄漏）

# —— 非零判定阈值（物理单位）用于统计与 NZ(true/pred) ——
NZ_THRESHOLD_ABS = 1e-4

# —— “高|F|”阈值：自动用训练集 |F| 的 p90（物理单位）；用于高值加权与欠估惩罚 ——
HIGHF_QUANTILE = 0.90

# —— 训练时的抽样目标（不删样本，用 WeightedRandomSampler）——
ZERO_TARGET_FRAC = 0.30  # 希望每个 epoch 里零样本占比（根据 NZ_THRESHOLD_ABS 判定）

# —— 损失：两种 loss 同时启用 + 欠估惩罚（阈值自适应）——
BETA_SMOOTHL1   = 0.1
USE_MAG_LOSS    = True
MAG_LOSS_WEIGHT = 0.5

USE_UNDER_PENALTY    = True
UNDER_PENALTY_WEIGHT = 1.0

# —— 损失内样本加权（默认先关，避免和 Sampler 双重叠加太猛；需要可以设 True）——
USE_LOSS_WEIGHT = False
W_NONZERO       = 2.0
W_HIGHF         = 4.0

# 早停
EARLYSTOP_PATIENCE = 20

# 输出
SAVE_DIR_BASE = "./artifacts_vec"

# GAM 配置（可选）
GAM_ENABLE = True          # 需要安装 pygam
GAM_MAX_SAMPLES = 200000   # 训练 GAM 的最大样本数（防止过大）
GAM_SPLINES = 10           # 每个特征的样条段数（terms）
GAM_LAM = 0.3              # 正则强度（pyGAM 的 lam）

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

def vec_mag(a):  # (N,3) -> (N,)
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
    n_val = max(1, int(round(len(files)*val_ratio_files)))
    val_files = files[-n_val:]
    tr = data[~data["__file__"].isin(val_files)].reset_index(drop=True)
    va = data[ data["__file__"].isin(val_files)].reset_index(drop=True)
    print(f"[INFO] Train rows={len(tr)} Val rows={len(va)} | files={len(files)-n_val}/{n_val}")
    return tr, va

def build_sequences_df(df, seq_len, stride, use_cols, target_cols):
    X = df[use_cols].to_numpy(dtype=np.float32)
    Y = df[target_cols].to_numpy(dtype=np.float32)
    T = len(df)
    if T<seq_len: return None
    xs,ys=[],[]
    for st in range(0, T-seq_len+1, stride):
        ed = st+seq_len
        xs.append(X[st:ed])
        ys.append(Y[ed-1])
    return np.stack(xs), np.stack(ys)

# ======================
# 模型 & 损失
# ======================
class GRUReg(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True,
                          dropout=(dropout if layers>1 else 0.0))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3)
        )
    def forward(self, x):
        out,_ = self.gru(x)         # [B,L,H]
        h = out[:, -1, :]           # 末步
        return self.head(h)         # [B,3]

class SeqDS(Dataset):
    def __init__(self, X, Y):
        self.X=X.astype(np.float32); self.Y=Y.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])

def compute_loss(y_pred, y_true, nz_th_scaled, high_th_scaled):
    # 分量 SmoothL1
    comp = F.smooth_l1_loss(y_pred, y_true, beta=BETA_SMOOTHL1, reduction='none').mean(dim=-1)  # (B,)
    total = comp

    # 模长损失
    if USE_MAG_LOSS:
        p_mag = torch.sqrt(torch.clamp((y_pred**2).sum(dim=-1), min=1e-18))
        t_mag = torch.sqrt(torch.clamp((y_true**2).sum(dim=-1), min=1e-18))
        mag_l = F.smooth_l1_loss(p_mag, t_mag, beta=BETA_SMOOTHL1, reduction='none')            # (B,)
        total = total + MAG_LOSS_WEIGHT * mag_l

    # 欠估惩罚（仅对高|F|且 pred<true）
    if USE_UNDER_PENALTY:
        if 'p_mag' not in locals():
            p_mag = torch.sqrt(torch.clamp((y_pred**2).sum(dim=-1), min=1e-18))
            t_mag = torch.sqrt(torch.clamp((y_true**2).sum(dim=-1), min=1e-18))
        under = torch.clamp(t_mag - p_mag, min=0.0)
        high  = (t_mag >= high_th_scaled).float()
        total = total + UNDER_PENALTY_WEIGHT * under * high

    # （可选）损失内样本加权
    if USE_LOSS_WEIGHT:
        if 't_mag' not in locals():
            t_mag = torch.sqrt(torch.clamp((y_true**2).sum(dim=-1), min=1e-18))
        w = torch.ones_like(t_mag)
        w = w * (1.0 + (W_NONZERO - 1.0) * (t_mag > nz_th_scaled).float())
        w = w * (1.0 + (W_HIGHF   - 1.0) * (t_mag >= high_th_scaled).float())
        w = w / w.mean().clamp_min(1e-6)
        total = total * w

    return total.mean()

# ======================
# 标定（向量模长）
# ======================
def linear_calibration(x_pred, y_true, nonneg=False):
    x = np.asarray(x_pred).reshape(-1,1)
    y = np.asarray(y_true).reshape(-1,1)
    X = np.concatenate([x, np.ones_like(x)], axis=1)
    lam = 1e-8
    coef = np.linalg.solve(X.T@X + lam*np.eye(2), X.T@y)
    a, b = float(coef[0,0]), float(coef[1,0])
    def apply(z):
        out = a*z + b
        if nonneg: out = np.maximum(out, 0.0)
        return out
    return a, b, apply

def apply_vector_mag_calib(Y_pred, a, b):
    mag = vec_mag(Y_pred)[:,None]
    mag_c = np.maximum(0.0, a*mag + b)
    scale = mag_c / (mag + 1e-12)
    return Y_pred * scale

# ======================
# 主流程
# ======================
def main():
    set_seed(); print(f"[INFO] device={device}")

    # 读数据
    df = load_all_csv(DATA_DIR)

    # === 修复信息泄露：先切分，再用“训练集”估计 y_scale 的 p95(|F|) ===
    tr_df, va_df = split_by_file(df, val_ratio_files=0.2)

    # 标准化特征（仅用训练集统计）
    mu = tr_df[FEATURES].mean().to_numpy()
    std = tr_df[FEATURES].std().to_numpy() + 1e-12
    def z(df_):
        X = df_[FEATURES].to_numpy()
        return (X - mu)/std
    trZ = tr_df.copy(); trZ[FEATURES] = z(tr_df)
    vaZ = va_df.copy(); vaZ[FEATURES] = z(va_df)

    # 序列（未缩放的 Y_abs，用于阈值/采样统计）
    tr_seq = build_sequences_df(trZ, SEQ_LEN, SEQ_STRIDE, FEATURES, TARGET_COLS)
    va_seq = build_sequences_df(vaZ, SEQ_LEN, SEQ_STRIDE, FEATURES, TARGET_COLS)
    assert tr_seq is not None and va_seq is not None, "序列长度不足"
    Xtr, Ytr_abs = tr_seq   # (N_tr, L, p), (N_tr, 3) 物理单位
    Xva, Yva_abs = va_seq

    # —— y_scale：仅用训练集 p95(|F|)（修复泄漏）——
    if Y_P95_SCALE is None:
        p95_abs = float(np.quantile(vec_mag(Ytr_abs), 0.95))
    else:
        p95_abs = float(Y_P95_SCALE)
    print(f"[INFO] p95(|F|) [train-only] = {p95_abs:.3e}")

    # —— 把 Y 缩放到训练单位 ——
    y_scale = max(p95_abs, 1e-8)
    Ytr = Ytr_abs / y_scale
    Yva = Yva_abs / y_scale

    # —— 自适应阈值（训练单位）——
    Fmag_tr_abs = vec_mag(Ytr_abs)
    nz_th_abs   = NZ_THRESHOLD_ABS
    high_th_abs = float(np.quantile(Fmag_tr_abs, HIGHF_QUANTILE))
    nz_th_scaled   = nz_th_abs / y_scale
    high_th_scaled = high_th_abs / y_scale

    print(f"[THRESH] NZ_THRESHOLD(abs)={nz_th_abs:.3e}, HIGHF_TH(abs,p{int(HIGHF_QUANTILE*100)})={high_th_abs:.3e}")
    print(f"[SCALE ] y_scale(p95 |F|)={y_scale:.3e} -> nz_th_scaled={nz_th_scaled:.3e}, high_th_scaled={high_th_scaled:.3e}")

    # —— WeightedRandomSampler（不删样本，仅改变抽样概率）——
    zero_mask = (Fmag_tr_abs <= nz_th_abs)
    raw_zero_frac = float(zero_mask.mean())
    p_zero = float(np.clip(ZERO_TARGET_FRAC, 1e-6, 1-1e-6))
    w_zero = p_zero / (raw_zero_frac + 1e-12)
    w_nz   = (1 - p_zero) / (1 - raw_zero_frac + 1e-12)
    # 额外给高|F|一点 boost（在非零里再放大）
    high_mask = (Fmag_tr_abs >= high_th_abs).astype(np.float32)
    weights = np.where(zero_mask, w_zero, w_nz) * (1.0 + 1.0 * high_mask)
    weights = torch.as_tensor(weights, dtype=torch.float32)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    print(f"[SAMPLE] raw_zero_frac={raw_zero_frac:.3f} -> target_zero_frac={p_zero:.3f} (使用加权采样，不删除样本)")

    # DataLoader
    train_loader = DataLoader(SeqDS(Xtr, Ytr), batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
    val_loader   = DataLoader(SeqDS(Xva, Yva), batch_size=BATCH_SIZE, shuffle=False)

    # 模型/优化
    model = GRUReg(len(FEATURES), hidden=128, layers=1, dropout=0.1).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 输出目录
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(SAVE_DIR_BASE)/f"GRU_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SAVE] -> {out_dir.resolve()}")

    # 训练 & 早停（看非零的 |F| R2）
    best = -1e9
    best_state = None
    no_imp = 0

    for ep in range(1, EPOCHS+1):
        model.train(); tot=0; n=0
        for xb, yb in train_loader:
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(set_to_none=True)
            yp = model(xb)
            loss = compute_loss(yp, yb, nz_th_scaled, high_th_scaled)
            loss.backward()
            if CLIP_GRAD>0: nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            opt.step()
            tot += float(loss.item())*xb.size(0); n += xb.size(0)

        # 验证（把输出反缩放回物理单位再评估）
        model.eval()
        YP_list = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                yp = model(xb)                 # 训练单位
                YP_list.append((yp.cpu().numpy()) * y_scale)  # 反缩放
        YP_abs = np.concatenate(YP_list, axis=0)

        # === 清理：移除无用/重复的 YT_abs 占位写法 ===
        YT_abs = Yva_abs  # 直接使用验证集真值（物理单位）

        # 评估指标
        magT = vec_mag(YT_abs)
        magP = vec_mag(YP_abs)
        r2_fx = r2_score_np(YT_abs[:,0], YP_abs[:,0])
        r2_fy = r2_score_np(YT_abs[:,1], YP_abs[:,1])
        r2_fz = r2_score_np(YT_abs[:,2], YP_abs[:,2])
        r2_all = r2_score_np(magT, magP)
        mask = (magT > NZ_THRESHOLD_ABS)
        r2_nz  = r2_score_np(magT[mask], magP[mask]) if mask.any() else float('nan')
        nz_true = float((magT>NZ_THRESHOLD_ABS).mean()); nz_pred = float((magP>NZ_THRESHOLD_ABS).mean())

        print(f"[Ep{ep:03d}] Loss={tot/max(1,n):.5f} | R2(Fx,Fy,Fz)=({r2_fx:.3f},{r2_fy:.3f},{r2_fz:.3f}) | "
              f"|F| R2_all={r2_all:.3f} R2_nz={r2_nz:.3f} | NZ(true/pred)={nz_true:.3f}/{nz_pred:.3f}")

        score = r2_nz
        if score > best + 1e-5:
            best = score; no_imp = 0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= EARLYSTOP_PATIENCE:
                print(f"[EarlyStopping] stop at epoch {ep}")
                break

    # 载入最佳
    if best_state is not None:
        model.load_state_dict(best_state)

    # 最终评估 & 画图（全部在“物理单位”下）
    model.eval()
    YP_abs = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            yp = model(xb).cpu().numpy() * y_scale
            YP_abs.append(yp)
    YP_abs = np.concatenate(YP_abs, axis=0)
    YT_abs = Yva_abs

    r2_fx = r2_score_np(YT_abs[:,0], YP_abs[:,0])
    r2_fy = r2_score_np(YT_abs[:,1], YP_abs[:,1])
    r2_fz = r2_score_np(YT_abs[:,2], YP_abs[:,2])
    magT = vec_mag(YT_abs); magP = vec_mag(YP_abs)
    r2_all = r2_score_np(magT, magP)
    mask = (magT > NZ_THRESHOLD_ABS)
    r2_nz  = r2_score_np(magT[mask], magP[mask]) if mask.any() else float('nan')
    print(f"[BEST] |F|: R2_all={r2_all:.3f}, R2_nz={r2_nz:.3f}")

    torch.save({"state_dict": model.state_dict(), "mu": tr_df[FEATURES].mean().to_numpy(),
                "std": tr_df[FEATURES].std().to_numpy()+1e-12, "features": FEATURES,
                "y_scale": y_scale},
               out_dir/"best_vecgru.pt")

    # 画散点（未校准）
    names = ["Fx","Fy","Fz"]
    for i,name in enumerate(names):
        plt.figure(figsize=(5,5))
        plt.scatter(YT_abs[:,i], YP_abs[:,i], s=8, alpha=0.35)
        lo = float(min(YT_abs[:,i].min(), YP_abs[:,i].min()))
        hi = float(max(YT_abs[:,i].max(), YP_abs[:,i].max()))
        plt.plot([lo,hi],[lo,hi],'k--',lw=1)
        plt.xlabel(f"True {name}"); plt.ylabel(f"Pred {name}")
        plt.title(f"VAL: {name} (R2={r2_score_np(YT_abs[:,i],YP_abs[:,i]):.3f})")
        plt.tight_layout(); plt.savefig(out_dir/f"pred_vs_true_{name}_VAL.png", dpi=160); plt.close()

    plt.figure(figsize=(5,5))
    lo = float(min(magT.min(), magP.min())); hi = float(max(magT.max(), magP.max()))
    plt.scatter(magT, magP, s=8, alpha=0.35)
    plt.plot([lo,hi],[lo,hi],'k--',lw=1)
    plt.xlabel("|F| True"); plt.ylabel("|F| Pred")
    plt.title(f"VAL: |F| (R2_all={r2_all:.3f}, R2_nz={r2_nz:.3f})")
    plt.tight_layout(); plt.savefig(out_dir/"pred_vs_true_Fmag_VAL.png", dpi=160); plt.close()

    plt.figure(figsize=(7,4))
    plt.hist(magP - magT, bins=80, alpha=0.9)
    plt.xlabel("Residual (|F|_pred - |F|_true)"); plt.ylabel("Count")
    plt.title("VAL: |F| Residual Histogram")
    plt.tight_layout(); plt.savefig(out_dir/"residual_hist_Fmag_VAL.png", dpi=160); plt.close()

    # 训练后标定：|F|（非负）
    a_m, b_m, fn_mag = linear_calibration(magP, magT, nonneg=True)
    magP_cal = fn_mag(magP)
    r2_all_c = r2_score_np(magT, magP_cal)
    mask = (magT > NZ_THRESHOLD_ABS)
    r2_nz_c  = r2_score_np(magT[mask], magP_cal[mask]) if mask.any() else float('nan')

    plt.figure(figsize=(5,5))
    plt.scatter(magT, magP_cal, s=8, alpha=0.35)
    plt.plot([lo,hi],[lo,hi],'k--',lw=1)
    plt.xlabel("|F| True"); plt.ylabel("|F| Pred (Calib)")
    plt.title(f"VAL: |F| Calibrated (R2_all={r2_all_c:.3f}, R2_nz={r2_nz_c:.3f})")
    plt.tight_layout(); plt.savefig(out_dir/"pred_vs_true_Fmag_VAL_CAL.png", dpi=160); plt.close()

    # 分量：只标定 Fy/Fz（Fx 保持原样）
    for i,(name,do_cal) in enumerate([("Fx",False),("Fy",True),("Fz",True)]):
        y_true = YT_abs[:,i]; y_pred = YP_abs[:,i]
        if do_cal:
            a,b,fn = linear_calibration(y_pred, y_true, nonneg=False)
            y_cal  = fn(y_pred)
            r2c = r2_score_np(y_true, y_cal)
            plt.figure(figsize=(5,5))
            lo = float(min(y_true.min(), y_cal.min())); hi=float(max(y_true.max(), y_cal.max()))
            plt.scatter(y_true, y_cal, s=8, alpha=0.35)
            plt.plot([lo,hi],[lo,hi],'k--',lw=1)
            plt.xlabel(f"True {name}"); plt.ylabel(f"Pred {name} (Calib)")
            plt.title(f"VAL: {name} (Calib R2={r2c:.3f})")
            plt.tight_layout(); plt.savefig(out_dir/f"pred_vs_true_{name}_VAL_CAL.png", dpi=160); plt.close()
        else:
            r2raw = r2_score_np(y_true, y_pred)
            plt.figure(figsize=(5,5))
            lo = float(min(y_true.min(), y_pred.min())); hi=float(max(y_true.max(), y_pred.max()))
            plt.scatter(y_true, y_pred, s=8, alpha=0.35)
            plt.plot([lo,hi],[lo,hi],'k--',lw=1)
            plt.xlabel(f"True {name}"); plt.ylabel(f"Pred {name}")
            plt.title(f"VAL: {name} (R2={r2raw:.3f})")
            plt.tight_layout(); plt.savefig(out_dir/f"pred_vs_true_{name}_VAL_CAL.png", dpi=160); plt.close()

    # 指标保存
    pd.DataFrame([{
        "R2_Fx": r2_fx, "R2_Fy": r2_fy, "R2_Fz": r2_fz,
        "R2_|F|_all_raw": r2_all, "R2_|F|_nz_raw": r2_nz,
        "R2_|F|_all_cal": r2_all_c, "R2_|F|_nz_cal": r2_nz_c,
        "NZ_true_ratio": float((magT>NZ_THRESHOLD_ABS).mean()),
        "NZ_pred_ratio": float((magP>NZ_THRESHOLD_ABS).mean()),
        "calib_|F|_a": a_m, "calib_|F|_b": b_m,
        "y_scale_p95": y_scale,
        "highF_abs_q": HIGHF_QUANTILE, "highF_abs_value": high_th_abs
    }]).to_csv(out_dir/"val_metrics.csv", index=False)

    print(f"[CALIB] |F|_calib = max(0, {a_m:.6f} * |F|_pred + {b_m:.6f})")

    # ======================
    # GAM（可解释性）: 用标准化特征 Z 拟合 -> 预测目标使用“物理单位” Fx/Fy/Fz 真值
    # ======================
    if GAM_ENABLE:
        try:
            from pygam import LinearGAM, s
            gam_dir = out_dir / "gam"
            gam_dir.mkdir(parents=True, exist_ok=True)
            print("[GAM] Training pyGAM interpretable models for Fx/Fy/Fz ...")

            # 准备数据（使用训练+验证的 Z, y_true_abs；也可只用训练集，按需改）
            # 这里选用训练集，避免用到验证标签做拟合解释 —— 更干净
            Z_tr = trZ[FEATURES].to_numpy(dtype=np.float32)
            Y_tr = tr_df[TARGET_COLS].to_numpy(dtype=np.float32)

            # 下采样（如样本超出阈值）
            if len(Z_tr) > GAM_MAX_SAMPLES:
                idx = np.random.choice(len(Z_tr), GAM_MAX_SAMPLES, replace=False)
                Z_tr = Z_tr[idx]; Y_tr = Y_tr[idx]

            for j, name in enumerate(["Fx","Fy","Fz"]):
                y = Y_tr[:, j]
                terms = sum([s(i, n_splines=GAM_SPLINES) for i in range(Z_tr.shape[1])])
                gam = LinearGAM(terms, lam=GAM_LAM).fit(Z_tr, y)

                # 保存每个特征的一维边际效应曲线
                figs = []
                for k, feat in enumerate(FEATURES):
                    XX = gam.generate_X_grid(term=k)
                    pdep = gam.partial_dependence(term=k, X=XX)
                    pd_df = pd.DataFrame({"feature_value": XX[:, k], f"PD_{name}": pdep})
                    pd_df.to_csv(gam_dir/f"pdp_{name}_{feat}.csv", index=False)

                    # 画图
                    plt.figure(figsize=(5,3.2))
                    plt.plot(XX[:, k], pdep, lw=2)
                    plt.title(f"GAM PDP: {name} vs {feat}")
                    plt.xlabel(feat); plt.ylabel(f"{name} (effect)")
                    plt.tight_layout(); plt.savefig(gam_dir/f"pdp_{name}_{feat}.png", dpi=150); plt.close()

                # 保存模型系数/summary
                try:
                    with open(gam_dir/f"summary_{name}.txt","w",encoding="utf-8") as f:
                        f.write(str(gam.summary()))
                except Exception as e:
                    warnings.warn(f"GAM summary write failed for {name}: {e}")

            print("[GAM] Done. Files saved under:", gam_dir)

        except Exception as e:
            warnings.warn(
                f"[GAM] pygam not available or failed: {e}\n"
                "Please install with `pip install pygam` if you want GAM plots. "
                "Main training results are already saved."
            )

    print("[DONE]")

if __name__ == "__main__":
    main()
