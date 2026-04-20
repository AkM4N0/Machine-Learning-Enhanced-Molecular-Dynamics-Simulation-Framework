# -*- coding: utf-8 -*-
import os, glob, random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader

###################################
# ========== 配置区 ==========
SAVE_DIR_BASE = "./artifacts_vec"
run_id = time.strftime("%Y%m%d-%H%M%S")
out_dir = Path(SAVE_DIR_BASE) / f"MLP_{run_id}"
out_dir.mkdir(parents=True, exist_ok=True)
BALANCE_USE = True
BALANCE_CHECK_COLS = ["U_int", "U_int_Fx", "U_int_Fy", "U_int_Fz", "U_int_Fmag"]
BALANCE_EPS = 0.0
BALANCE_RANDOM_OFFSET = 0
BALANCE_BY_GROUP = False
TIME_COL = None

DATA_DIR = "nn_input_data_xy"
GLOB_PAT = "*.csv"

FEATURES = [
    "d","r0","h0",
    "pair_0_1_qrel_x","pair_0_1_qrel_y","pair_0_1_qrel_z","pair_0_1_qrel_w"
]
TARGET_COLS = ["U_int_Fx","U_int_Fy","U_int_Fz"]

SEQ_LEN = 5
BATCH_SIZE = 512
EPOCHS = 50
LR = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLS_WEIGHT = 0.5
REG_WEIGHT = 1.5
EPS_P = 1e-3

USE_DIR_LOSS = True
DIR_LOSS_WEIGHT = 0.3

###################################
# ========== 工具 / 数据处理 ==========

def set_seed(s=2025):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def vec_mag(a):
    return np.sqrt(np.clip((a * a).sum(axis=-1), 1e-18, None))

def fix_quat_block(df, prefix="pair_0_1_qrel"):
    cols = [f"{prefix}_{k}" for k in ("x","y","z","w")]
    q = df[cols].astype(float).to_numpy()
    n = np.linalg.norm(q, axis=1, keepdims=True)
    bad = (n.squeeze() < 1e-9) | ~np.isfinite(n.squeeze())
    qn = np.ones_like(q)
    qn[:,3] = 1.0
    good = ~bad
    qn[good] = q[good] / (n[good] + 1e-12)
    flip = qn[:,3] < 0
    qn[flip] = -qn[flip]
    df.loc[:, cols] = qn

def add_quat_dynamics(df, prefix="pair_0_1_qrel"):
    x,y,z,w = [f"{prefix}_{k}" for k in ("x","y","z","w")]
    if not all(c in df.columns for c in (x,y,z,w)):
        return df
    Q = df[[x,y,z,w]].to_numpy(float)
    n = np.linalg.norm(Q, axis=1, keepdims=True)
    n[n==0] = 1.0
    Qn = Q / n
    L = len(Qn)
    ang_full = np.zeros((L,), dtype=float)
    if L >= 2:
        dot = np.sum(Qn[1:] * Qn[:-1], axis=1)
        dot = np.clip(np.abs(dot), -1.0, 1.0)
        ang = 2.0 * np.degrees(np.arccos(dot))
        ang_full[1:] = ang
    col = f"{prefix}_dAng_deg"
    df[col] = ang_full
    if col not in FEATURES:
        FEATURES.append(col)
    return df

def load_and_balance_csv(data_dir, time_col=TIME_COL) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, GLOB_PAT)))
    assert files, f"未找到数据：{data_dir}/{GLOB_PAT}"
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["__file__"] = os.path.basename(f)
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

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

    miss = [c for c in BALANCE_CHECK_COLS if c not in all_df.columns]
    if miss:
        raise KeyError(f"缺少用于判零的列: {miss}")
    is_all_zero = all_df[BALANCE_CHECK_COLS].abs().le(BALANCE_EPS).all(axis=1)

    def _balance_pair(df_sub):
        df_zero = df_sub.loc[ is_all_zero[df_sub.index] ].copy()
        df_nonz = df_sub.loc[~is_all_zero[df_sub.index] ].copy()
        Nz, Nnz = len(df_zero), len(df_nonz)
        if Nnz == 0:
            df_zero_sys = df_zero.iloc[0:0].copy()
        elif Nz == 0:
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
        parts = []
        for _, g in all_df.groupby("__file__", sort=False):
            parts.append(_balance_pair(g))
        bal = pd.concat(parts, ignore_index=True)
    else:
        bal = _balance_pair(all_df)

    fix_quat_block(bal, "pair_0_1_qrel")
    bal = add_quat_dynamics(bal, prefix="pair_0_1_qrel")

    need_cols = FEATURES + TARGET_COLS + ["__file__"]
    miss2 = [c for c in need_cols if c not in bal.columns]
    if miss2:
        raise KeyError(f"缺少训练所需列: {miss2}")
    sub = bal[need_cols].copy()
    print(f"[BAL] 最终样本数: {len(sub)}（全局）")
    return sub

def split_by_file(df, val_ratio=0.2):
    files = sorted(df["__file__"].unique())
    nval = max(1, int(len(files) * val_ratio))
    val_files = files[-nval:]
    tr = df[~df["__file__"].isin(val_files)].reset_index(drop=True)
    va = df[df["__file__"].isin(val_files)].reset_index(drop=True)
    return tr, va

def build_sequences(df, seq_len):
    Xs, Ys = [], []
    N = len(df)
    arrX = df[FEATURES].to_numpy(dtype=float)
    arrY = df[TARGET_COLS].to_numpy(dtype=float)
    for i in range(N - seq_len + 1):
        Xs.append(arrX[i:i+seq_len])
        Ys.append(arrY[i+seq_len-1])
    if not Xs:
        return None
    return np.stack(Xs), np.stack(Ys)

class SeqDS(Dataset):
    def __init__(self, X, Y):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])

###################################
# ========== 模型 + 损失 ==========

class MultiTaskMLPReg(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_dims=[256,128], dropout=0.1):
        super().__init__()
        in_dim = seq_len * feat_dim
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        self.net = nn.Sequential(*layers)
        self.class_head = nn.Linear(prev, 1)
        self.reg_head = nn.Linear(prev, 3)
        self.alpha = nn.Parameter(torch.ones(3) * 0.1)
        self.beta = nn.Parameter(torch.zeros(3))
        self.use_softplus = True

    def forward(self, x):
        B, L, F_shap = x.shape
        x_flat = x.view(B, L * F_shap)
        h = self.net(x_flat)
        logit = self.class_head(h).squeeze(-1)
        p = torch.sigmoid(logit)
        y_reg = self.reg_head(h)
        alpha = F.softplus(self.alpha) if self.use_softplus else self.alpha
        y_cal = y_reg * alpha + self.beta
        y_final = y_cal * p.unsqueeze(-1)
        return p, y_final

def compute_loss_multitask(p, y_cal, y_true, nz_th_scaled):
    mag_t = torch.sqrt(torch.clamp((y_true**2).sum(dim=-1), min=1e-18))
    is_nonzero = (mag_t > nz_th_scaled).float()
    cls_loss = F.binary_cross_entropy(p, is_nonzero)
    loss_reg = F.smooth_l1_loss(y_cal, y_true, reduction='none').mean(dim=-1)
    weighted_reg = (loss_reg * (p + EPS_P)).mean()
    loss = CLS_WEIGHT * cls_loss + REG_WEIGHT * weighted_reg
    if USE_DIR_LOSS:
        dot = torch.sum(y_cal * y_true, dim=-1)
        mag_p = torch.sqrt(torch.clamp((y_cal**2).sum(dim=-1), min=1e-18))
        cos_sim = dot / (mag_p * mag_t + 1e-12)
        dir_loss = (1.0 - cos_sim).clamp(min=0.0)
        loss = loss + DIR_LOSS_WEIGHT * dir_loss.mean()
    return loss, cls_loss.item(), weighted_reg.item()

###################################
# ========== 训练 + 早停 + 分阶段 ==========

def train_with_early_stopping(model, train_loader, val_loader, optimizer,
                              nz_th_scaled, max_epochs=50, patience=5):
    best_metric = None
    best_state = None
    no_improve = 0

    for ep in range(1, max_epochs + 1):
        model.train()
        tot_loss = tot_cls = tot_reg = cnt = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            p, y_cal = model(xb)
            loss, cls_l, reg_l = compute_loss_multitask(p, y_cal, yb, nz_th_scaled)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * xb.size(0)
            tot_cls += cls_l * xb.size(0)
            tot_reg += reg_l * xb.size(0)
            cnt += xb.size(0)
        avg_loss = tot_loss / cnt

        model.eval()
        YP_list = []
        YT_list = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                p, y_cal = model(xb)
                YP_list.append(y_cal.cpu().numpy())
                YT_list.append(yb.numpy())
        YP = np.concatenate(YP_list, axis=0)
        YT = np.concatenate(YT_list, axis=0)

        magT = np.linalg.norm(YT, axis=1)
        magP = np.linalg.norm(YP, axis=1)
        ssr = ((magT - magP)**2).sum()
        sst = ((magT - magT.mean())**2).sum() + 1e-12
        r2_all = 1.0 - ssr / sst

        print(f"[Val] Epoch {ep} | R2|F| = {r2_all:.5f} | loss = {avg_loss:.5f}")

        if (best_metric is None) or (r2_all > best_metric):
            best_metric = r2_all
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at epoch {ep}. Best R2|F| = {best_metric:.5f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_metric

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

def _linear_calibration(x_pred, y_true, nonneg=False):
    """
    最小二乘线性标定 y ≈ a*x + b
    返回: a, b, apply_fn(z)=max(0, a*z+b)（当 nonneg=True 时做非负截断）
    """
    x = np.asarray(x_pred).reshape(-1, 1)
    y = np.asarray(y_true).reshape(-1, 1)
    X = np.concatenate([x, np.ones_like(x)], axis=1)
    lam = 1e-8  # 极小L2正则，防奇异
    a, b = np.linalg.solve(X.T @ X + lam * np.eye(2), X.T @ y).ravel()

    def apply_fn(z):
        out = a * z + b
        return np.maximum(out, 0.0) if nonneg else out

    return a, b, apply_fn

def visualize_like_gru(YT, YP, va_df, SEQ_LEN, save_dir_base="./artifacts_vec", tag="MLP"):
    names = ["Fx","Fy","Fz"]
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(save_dir_base) / f"{tag}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 分量 Pred vs True
    for i, name in enumerate(names):
        y_true, y_pred = YT[:, i], YP[:, i]
        lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
        plt.figure(figsize=(5,5))
        plt.scatter(y_true, y_pred, s=8, alpha=0.35)
        plt.plot([lo,hi], [lo,hi], 'k--', lw=1)
        plt.xlabel(f"True {name}"); plt.ylabel(f"Pred {name}")
        plt.title(f"VAL: {name} (R2={r2_score(y_true,y_pred):.3f})")
        plt.tight_layout(); plt.savefig(out_dir / f"pred_vs_true_{name}_VAL.png", dpi=160); plt.close()

    # 2) |F| 模长散点 + 残差直方图
    magT = np.linalg.norm(YT, axis=1)
    magP = np.linalg.norm(YP, axis=1)
    lo, hi = float(min(magT.min(), magP.min())), float(max(magT.max(), magP.max()))
    plt.figure(figsize=(5,5))
    plt.scatter(magT, magP, s=8, alpha=0.35)
    plt.plot([lo,hi], [lo,hi], 'k--', lw=1)
    plt.xlabel("|F| True"); plt.ylabel("|F| Pred")
    plt.title(f"VAL: |F| (R2_all={r2_score(magT,magP):.3f})")
    plt.tight_layout(); plt.savefig(out_dir / "pred_vs_true_Fmag_VAL.png", dpi=160); plt.close()

    plt.figure(figsize=(7,4))
    plt.hist(magP - magT, bins=80, alpha=0.9)
    plt.xlabel("Residual (|F|_pred - |F|_true)"); plt.ylabel("Count")
    plt.title("VAL: |F| Residual Histogram")
    plt.tight_layout(); plt.savefig(out_dir / "residual_hist_Fmag_VAL.png", dpi=160); plt.close()

    # 3) |F| 线性标定
    a_m, b_m, fn_mag = _linear_calibration(magP, magT, nonneg=True)
    magP_cal = fn_mag(magP)
    plt.figure(figsize=(5,5))
    plt.scatter(magT, magP_cal, s=8, alpha=0.35)
    plt.plot([lo,hi], [lo,hi], 'k--', lw=1)
    plt.xlabel("|F| True"); plt.ylabel("|F| Pred (Calib)")
    plt.title(f"VAL: |F| Calibrated (R2_all={r2_score(magT,magP_cal):.3f})")
    plt.tight_layout(); plt.savefig(out_dir / "pred_vs_true_Fmag_VAL_CAL.png", dpi=160); plt.close()
    print(f"[CALIB] |F|_calib = max(0, {a_m:.6f} * |F|_pred + {b_m:.6f})")

    # 4) 分量线性标定（Fx略，Fy/Fz标定，与 GRU 脚本一致）
    for i, (name, do_cal) in enumerate([("Fx", False), ("Fy", True), ("Fz", True)]):
        y_true, y_pred = YT[:, i], YP[:, i]
        if do_cal:
            a, b, fn = _linear_calibration(y_pred, y_true, nonneg=False)
            y_cal = fn(y_pred)
            lo, hi = float(min(y_true.min(), y_cal.min())), float(max(y_true.max(), y_cal.max()))
            plt.figure(figsize=(5,5))
            plt.scatter(y_true, y_cal, s=8, alpha=0.35)
            plt.plot([lo,hi], [lo,hi], 'k--', lw=1)
            plt.xlabel(f"True {name}"); plt.ylabel(f"Pred {name} (Calib)")
            plt.title(f"VAL: {name} (Calib R2={r2_score(y_true,y_cal):.3f})")
            plt.tight_layout(); plt.savefig(out_dir / f"pred_vs_true_{name}_VAL_CAL.png", dpi=160); plt.close()
        else:
            lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
            plt.figure(figsize=(5,5))
            plt.scatter(y_true, y_pred, s=8, alpha=0.35)
            plt.plot([lo,hi], [lo,hi], 'k--', lw=1)
            plt.xlabel(f"True {name}"); plt.ylabel(f"Pred {name}")
            plt.title(f"VAL: {name} (R2={r2_score(y_true,y_pred):.3f})")
            plt.tight_layout(); plt.savefig(out_dir / f"pred_vs_true_{name}_VAL_CAL.png", dpi=160); plt.close()

    # 5) 按距离 d 分箱
    d_full = va_df["d"].to_numpy()
    offset = SEQ_LEN - 1
    d_va = d_full[offset: offset + len(YT)] if len(d_full) != len(YT) else d_full
    bins = np.linspace(d_va.min(), d_va.max(), 11)
    idx = np.digitize(d_va, bins)
    rows = []
    for b in range(1, len(bins)):
        m = (idx == b)
        if m.sum() < 20: continue
        segT, segP = YT[m], YP[m]
        segMagT, segMagP = np.linalg.norm(segT,axis=1), np.linalg.norm(segP,axis=1)
        rows.append({"bin":b,"d_min":bins[b-1],"d_max":bins[b],
                     "count":int(m.sum()),
                     "MAE_|F|":mean_absolute_error(segMagT,segMagP),
                     "RMSE_|F|":np.sqrt(mean_squared_error(segMagT,segMagP)),
                     "R2_|F|":r2_score(segMagT,segMagP),
                     "MAE_Fx":mean_absolute_error(segT[:,0],segP[:,0]),
                     "MAE_Fy":mean_absolute_error(segT[:,1],segP[:,1]),
                     "MAE_Fz":mean_absolute_error(segT[:,2],segP[:,2]),
                     "Pearson_r_|F|_vs_d": pearsonr(segMagT, d_va[m])[0] if m.sum()>2 else np.nan})
    df_bins = pd.DataFrame(rows)
    df_bins.to_csv(out_dir/"bin_error_vs_distance_d.csv", index=False)
    mid = (df_bins["d_min"] + df_bins["d_max"]) / 2
    plt.figure(figsize=(6,4))
    plt.plot(mid, df_bins["MAE_|F|"], marker='o', label="MAE(|F|)")
    plt.plot(mid, df_bins["RMSE_|F|"], marker='s', linestyle='--', label="RMSE(|F|)")
    plt.xlabel("Distance d"); plt.ylabel("Error (|F|)"); plt.title("Error vs Distance d")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_dir/"error_vs_distance_d.png", dpi=160); plt.close()

    # 6) 按姿态角差 φ 分箱（由相对四元数 w 分量求）
    q_cols = ["pair_0_1_qrel_x","pair_0_1_qrel_y","pair_0_1_qrel_z","pair_0_1_qrel_w"]
    q = va_df[q_cols].to_numpy(float)
    theta = 2.0 * np.arccos(np.clip(np.abs(q[:,3]), 0.0, 1.0))  # rad → deg
    phi_deg = np.degrees(theta)
    phi_use = phi_deg[offset: offset + len(YT)] if len(phi_deg) != len(YT) else phi_deg
    bins_phi = np.linspace(phi_use.min(), phi_use.max(), 11)
    idx_phi = np.digitize(phi_use, bins_phi)
    rows_phi = []
    for b in range(1, len(bins_phi)):
        m = (idx_phi == b)
        if m.sum() < 20: continue
        segT, segP = YT[m], YP[m]
        segMagT, segMagP = np.linalg.norm(segT,axis=1), np.linalg.norm(segP,axis=1)
        rows_phi.append({"bin_phi":b,"phi_lo_deg":bins_phi[b-1],"phi_hi_deg":bins_phi[b],
                         "count":int(m.sum()),
                         "MAE_|F|":mean_absolute_error(segMagT,segMagP),
                         "RMSE_|F|":np.sqrt(mean_squared_error(segMagT,segMagP)),
                         "R2_|F|":r2_score(segMagT,segMagP),
                         "MAE_Fx":mean_absolute_error(segT[:,0],segP[:,0]),
                         "MAE_Fy":mean_absolute_error(segT[:,1],segP[:,1]),
                         "MAE_Fz":mean_absolute_error(segT[:,2],segP[:,2]),
                         "Pearson_r_|F|_vs_phi": pearsonr(segMagT, phi_use[m])[0] if m.sum()>2 else np.nan})
    df_phi = pd.DataFrame(rows_phi)
    df_phi.to_csv(out_dir/"bin_error_vs_angle_phi.csv", index=False)
    mid_phi = (df_phi["phi_lo_deg"] + df_phi["phi_hi_deg"]) / 2
    plt.figure(figsize=(6,4))
    plt.plot(mid_phi, df_phi["MAE_|F|"], marker='o', label="MAE(|F|)")
    plt.plot(mid_phi, df_phi["RMSE_|F|"], marker='s', linestyle='--', label="RMSE(|F|)")
    plt.xlabel("Orientation angle difference φ (deg)"); plt.ylabel("Error (|F|)")
    plt.title("Error vs Orientation φ")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_dir/"error_vs_angle_phi.png", dpi=160); plt.close()

    print(f"[SAVE] 可视化已输出至: {out_dir.resolve()}")


def main():
    import numpy as np
    import matplotlib as plt
    set_seed()
    df = load_and_balance_csv(DATA_DIR, time_col=TIME_COL)
    tr, va = split_by_file(df, 0.2)

    # 标签标准化
    yt_tr = tr[TARGET_COLS].to_numpy(dtype=float)
    yt_mean = yt_tr.mean(axis=0)
    yt_std = yt_tr.std(axis=0) + 1e-12
    tr[TARGET_COLS] = (tr[TARGET_COLS].to_numpy() - yt_mean) / yt_std
    va[TARGET_COLS] = (va[TARGET_COLS].to_numpy() - yt_mean) / yt_std

    seq_tr = build_sequences(tr, SEQ_LEN)
    seq_va = build_sequences(va, SEQ_LEN)
    if seq_tr is None or seq_va is None:
        raise RuntimeError("序列长度不足")

    Xtr, Ytr = seq_tr
    Xva, Yva = seq_va

    train_loader = DataLoader(SeqDS(Xtr, Ytr), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(SeqDS(Xva, Yva), batch_size=BATCH_SIZE, shuffle=False)

    model = MultiTaskMLPReg(SEQ_LEN, len(FEATURES), hidden_dims=[256,128]).to(device)

    # 阶段 1：只训练分类头（冻结回归头、alpha、beta）
    for param in model.reg_head.parameters():
        param.requires_grad = False
    model.alpha.requires_grad = False
    model.beta.requires_grad = False

    optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    print("=== Phase 1: Training classification head only ===")
    train_with_early_stopping(model, train_loader, val_loader, optimizer1,
                              nz_th_scaled=1e-3, max_epochs=10, patience=3)

    # 阶段 2：解冻回归头 & alpha & beta，共同训练
    for param in model.reg_head.parameters():
        param.requires_grad = True
    model.alpha.requires_grad = True
    model.beta.requires_grad = True

    optimizer2 = torch.optim.Adam(model.parameters(), lr=LR * 0.5)
    print("=== Phase 2: Joint training regression + classification ===")
    train_with_early_stopping(model, train_loader, val_loader, optimizer2,
                              nz_th_scaled=1e-3, max_epochs=EPOCHS, patience=5)

    # 验证 / 输出最终评价（反标准化后评价）
    model.eval()
    YP_norm = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            p, y_cal = model(xb)
            YP_norm.append(y_cal.cpu().numpy())
    YP_norm = np.concatenate(YP_norm, axis=0)
    YP = YP_norm * yt_std[None, :] + yt_mean[None, :]
    YT = (Yva * yt_std[None, :] + yt_mean[None, :])
    visualize_like_gru(YT, YP, va, SEQ_LEN, save_dir_base="./artifacts_vec", tag="MLP")


    magT = vec_mag(YT)
    magP = vec_mag(YP)
    def r2(y_true, y_pred):
        ssr = ((y_true - y_pred)**2).sum()
        sst = ((y_true - y_true.mean())**2).sum() + 1e-12
        return 1.0 - ssr / sst

    r2_fx = r2(YT[:,0], YP[:,0])
    r2_fy = r2(YT[:,1], YP[:,1])
    r2_fz = r2(YT[:,2], YP[:,2])
    r2_all = r2(magT, magP)
    mask = (magT > 1e-6)
    r2_nz = r2(magT[mask], magP[mask]) if mask.any() else float("nan")

    print(f"=== Final Eval === R2 Fx,Fy,Fz=({r2_fx:.3f},{r2_fy:.3f},{r2_fz:.3f}) | R2|F| all={r2_all:.3f} nz={r2_nz:.3f}")

if __name__ == "__main__":
    main()
