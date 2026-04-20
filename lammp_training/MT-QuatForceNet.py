# -*- coding: utf-8 -*-
"""
train_GRU_fy_focus.py
最小可运行版：分层切分 + WeightedRandomSampler + Fy分类回归双头 + 稳定日志
"""
import os, glob, time, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --------------------------
# 配置
# --------------------------
DATA_DIR   = "nn_input_data_all"   # 你的 CSV 目录
GLOB_PAT   = "*.csv"
SEED       = 42

# 输入与目标列名
FEATURES = [
    "d","r0","h0",
    "pair_0_1_qrel_x","pair_0_1_qrel_y","pair_0_1_qrel_z","pair_0_1_qrel_w"
]
TARGET_COLS = ["U_int_Fx","U_int_Fy","U_int_Fz"]

# 阈值（绝对值判定“非零力”）
NZ_THRESHOLD_ABS   = 0.665   # |F|>该值视为非零（评估阈值）
HIGHF_FIXED_ABS    = 6.447   # 强力阈值（可用于诊断）

# 采样与训练
BATCH_SIZE = 256
NUM_WORKERS = 2
EPOCHS = 200
PATIENCE = 20
LR = 2e-3
WD = 1e-4
CLIP_NORM = 1.0

# 训练集目标：期望 batch 内 Fy 非零占比（通过 WeightedRandomSampler 逼近）
TARGET_FY_NZ_RATIO_IN_BATCH = 0.35

# 最多训练样本限制（避免一次性吃满 140 万行；会保留全部非零 Fy，再有策略地抽零）
MAX_TRAIN_SAMPLES = 0  # 视显存/硬件可调；设 0 表示不限制


# --------------------------
# 工具
# --------------------------
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_all_csv_rows(data_dir, pattern):
    paths = sorted(glob.glob(str(Path(data_dir) / pattern)))
    if not paths:
        raise FileNotFoundError(f"No CSV found under {data_dir} with pattern {pattern}")
    dfs = []
    file_id = []
    for i, p in enumerate(paths):
        df = pd.read_csv(p)
        # 只保留必要列（避免额外内存）
        missing = set(FEATURES + TARGET_COLS) - set(df.columns)
        if missing:
            raise KeyError(f"Missing columns in {p}: {missing}")
        df = df[FEATURES + TARGET_COLS].copy()
        dfs.append(df)
        file_id.append(np.full(len(df), i, dtype=np.int32))
    big = pd.concat(dfs, ignore_index=True)
    file_id = np.concatenate(file_id, axis=0)
    return big, file_id, paths

def compute_force_mag(Y):
    return np.linalg.norm(Y, axis=1)

def zscore_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    return mu, std

def zscore_apply(X, mu, std):
    return (X - mu) / std

def r2_np(y_true, y_pred):
    # y_true, y_pred: (N,) 或 (N,3)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    ss_res = np.sum((y_true - y_pred)**2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0))**2, axis=0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return r2

def plot_val_scatters(model, val_dl, device, outdir="./output/scatter_val", max_points=20000):
    os.makedirs(outdir, exist_ok=True)

    Ys_true, Ys_pred = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device); yb = yb.to(device)
            yp = model(xb)

            # 兼容 tuple / dict 等返回
            if isinstance(yp, (tuple, list)):
                yp = yp[0]
            elif isinstance(yp, dict):
                yp = yp.get("vec", next(iter(yp.values())))

            Ys_true.append(yb.detach().cpu().numpy())  # (B,3)
            Ys_pred.append(yp.detach().cpu().numpy())  # (B,3)

    # 关键：拼成 (N,3)
    Y_true = np.concatenate(Ys_true, axis=0)
    Y_pred = np.concatenate(Ys_pred, axis=0)

    def _plot_one(y_true, y_pred, title, path):
        n = len(y_true)
        if n > max_points:
            idx = np.random.RandomState(42).choice(n, max_points, replace=False)
            y_true = y_true[idx]; y_pred = y_pred[idx]

        lim = np.percentile(np.abs(np.concatenate([y_true, y_pred])), 99.5)
        lim = float(max(lim, 1e-6))

        plt.figure(figsize=(5, 5), dpi=150)
        plt.scatter(y_true, y_pred, s=4, alpha=0.5)
        plt.plot([-lim, lim], [-lim, lim], "k--", linewidth=1)
        plt.xlabel("True"); plt.ylabel("Predicted"); plt.title(title)
        plt.tight_layout(); plt.savefig(path); plt.close()

    _plot_one(Y_true[:, 0], Y_pred[:, 0], "Fx: Pred vs True", os.path.join(outdir, "Fx_scatter.png"))
    _plot_one(Y_true[:, 1], Y_pred[:, 1], "Fy: Pred vs True", os.path.join(outdir, "Fy_scatter.png"))
    _plot_one(Y_true[:, 2], Y_pred[:, 2], "Fz: Pred vs True", os.path.join(outdir, "Fz_scatter.png"))

    print(f"[PLOT] 散点图已保存到 {outdir} (Fx_scatter.png / Fy_scatter.png / Fz_scatter.png)")

def _unwrap_vec(out):
    """将 model(x) 的输出统一成 [B,3] 的向量预测张量。"""
    # tuple/list: 取第一个作为向量
    if isinstance(out, (tuple, list)):
        out = out[0]
    # dict: 优先取 'vec'，否则取第一个 value
    if isinstance(out, dict):
        out = out.get('vec', next(iter(out.values())))
    return out  # 期望是 [B,3] 张量

def _nice_limits(y_true, y_pred, q_low=1, q_high=99, margin=0.06):
    a = np.concatenate([y_true, y_pred])
    lo = np.percentile(a, q_low)
    hi = np.percentile(a, q_high)
    span = hi - lo
    if span <= 1e-8:
        span = max(1.0, abs(hi)) * 0.1
    lo -= span * margin
    hi += span * margin
    return lo, hi

def _fit_line(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    y_hat = a * x + b
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return a, b, r2

def _stratified_sample(x, y, max_points=4000, bins=40, seed=42):
    n = len(x)
    if (max_points is None) or (n <= max_points):
        return x, y
    rng = np.random.default_rng(seed)
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    keep_idx = []
    quota = max(1, max_points // bins)
    for i in range(bins):
        mask = (x >= edges[i]) & (x <= edges[i + 1])
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        if len(idx) > quota:
            idx = rng.choice(idx, size=quota, replace=False)
        keep_idx.extend(idx.tolist())
    keep_idx = np.array(keep_idx, dtype=int)
    return x[keep_idx], y[keep_idx]

def _scatter_or_hex(x, y, ax, use_hex=False):
    if use_hex:
        hb = ax.hexbin(x, y, gridsize=40, bins='log', linewidths=0.0)
        cb = plt.colorbar(hb, ax=ax, pad=0.01, fraction=0.046)
        cb.set_label("log(count)")
    else:
        ax.scatter(x, y, s=18, alpha=0.45, edgecolors='none')

def pretty_scatter(y_true, y_pred, title, outfile=None,
                   max_points=4000, use_hex_if_big=True):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    x, y = _stratified_sample(y_true, y_pred, max_points=max_points)

    lo, hi = _nice_limits(x, y, q_low=1, q_high=99, margin=0.06)

    fig = plt.figure(figsize=(6.2, 6.2), dpi=140)
    ax = plt.gca()
    use_hex = (use_hex_if_big and len(x) > 8000)
    _scatter_or_hex(x, y, ax, use_hex=use_hex)

    ax.plot([lo, hi], [lo, hi], ls='--', lw=1.5, alpha=0.7)

    a, b, r2 = _fit_line(x, y)
    xs = np.linspace(lo, hi, 200)
    ax.plot(xs, a * xs + b, lw=1.6, alpha=0.9)
    ax.text(0.04, 0.96, f"$y={a:.2f}x{b:+.2f}$\n$R^2={r2:.3f}$",
            transform=ax.transAxes, ha='left', va='top')

    ax.set_title(title, pad=10)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, ls=':', alpha=0.35)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.tight_layout()

    if outfile:
        plt.savefig(outfile, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_val_scatters_pretty(model, val_dl, device, outdir="./output",
                             max_points=4000, use_hex_if_big=True):
    os.makedirs(outdir, exist_ok=True)
    Ys_true, Ys_pred = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device); yb = yb.to(device)
            out = model(xb)
            vec = _unwrap_vec(out)                # <—— 关键：稳健解包
            Ys_true.append(yb.detach().cpu().numpy())
            Ys_pred.append(vec.detach().cpu().numpy())
    Ys_true = np.concatenate(Ys_true, axis=0)  # (N,3)
    Ys_pred = np.concatenate(Ys_pred, axis=0)  # (N,3)

    pretty_scatter(Ys_true[:,0], Ys_pred[:,0], "Fx: Pred vs True",
                   os.path.join(outdir, "Fx_scatter_pretty.png"),
                   max_points=max_points, use_hex_if_big=use_hex_if_big)
    pretty_scatter(Ys_true[:,1], Ys_pred[:,1], "Fy: Pred vs True",
                   os.path.join(outdir, "Fy_scatter_pretty.png"),
                   max_points=max_points, use_hex_if_big=use_hex_if_big)
    pretty_scatter(Ys_true[:,2], Ys_pred[:,2], "Fz: Pred vs True",
                   os.path.join(outdir, "Fz_scatter_pretty.png"),
                   max_points=max_points, use_hex_if_big=use_hex_if_big)
    print(f"[PLOT] 漂亮版散点图已保存到 {outdir} (*_scatter_pretty.png)")


class ForceDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X; self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.Y[i]

# --------------------------
# 模型：共享 GRU 干路 + 两个头
#   - reg_head: 输出 Fx,Fy,Fz 回归
#   - cls_head: 输出 Fy 非零 logits（二分类）
# --------------------------
class GRU_FY_TwoHead(nn.Module):
    def __init__(self, in_dim, hidden=256, layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers,
                          batch_first=True, dropout=(dropout if layers>1 else 0.0))
        self.norm = nn.LayerNorm(hidden)

        self.reg_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 3)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)  # Fy-nonzero logits
        )

    def forward(self, x):
        if x.dim()==2:
            x = x.unsqueeze(1)  # (B,1,F)
        out, _ = self.gru(x)
        h = self.norm(out[:, -1, :])
        vec = self.reg_head(h)         # (B,3)
        fy_logits = self.cls_head(h)   # (B,1)
        return vec, fy_logits.squeeze(-1)  # (B,3), (B,)


# --------------------------
# 评估
# --------------------------
@torch.no_grad()
def evaluate(model, loader, device, nz_th_abs=NZ_THRESHOLD_ABS):
    model.eval()
    preds = []
    trues = []
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        vp, _ = model(xb)
        preds.append(vp.detach().cpu().numpy())
        trues.append(yb.detach().cpu().numpy())
    Yp = np.concatenate(preds, axis=0)
    Yt = np.concatenate(trues, axis=0)

    r2 = r2_np(Yt, Yp)  # (3,)
    Fx_r2, Fy_r2, Fz_r2 = float(r2[0]), float(r2[1]), float(r2[2])

    mag_true = np.linalg.norm(Yt, axis=1)
    mag_pred = np.linalg.norm(Yp, axis=1)
    r2_all = float(r2_np(mag_true, mag_pred)[0])

    nz_true = float((np.abs(Yt[:,1]) > nz_th_abs).mean())  # 只看 Fy 的非零比例（方便对比）
    nz_pred = float((np.abs(Yp[:,1]) > nz_th_abs).mean())

    # 非零子集上的 |F| R²（仅统计 Fy 非零的样本上整体 |F|）
    mask_nz = (mag_true > nz_th_abs)
    if mask_nz.any():
        r2_nz = float(r2_np(mag_true[mask_nz], mag_pred[mask_nz])[0])
    else:
        r2_nz = float('nan')

    return Fx_r2, Fy_r2, Fz_r2, r2_all, r2_nz, nz_true, nz_pred


# --------------------------
# 主流程
# --------------------------
def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device.type}")
    print("[INFO] Loading data...")

    df, file_id, files = read_all_csv_rows(DATA_DIR, GLOB_PAT)
    X_all = df[FEATURES].to_numpy(dtype=np.float32)
    Y_all = df[TARGET_COLS].to_numpy(dtype=np.float32)

    Fy_abs = np.abs(Y_all[:,1])
    fy_nz_flag = (Fy_abs > NZ_THRESHOLD_ABS).astype(np.int32)
    Fmag_all = compute_force_mag(Y_all)

    print("\n[FY ANALYSIS]")
    print(f"  Total samples: {len(X_all):,}")
    print(f"  Non-zero Fy samples: {fy_nz_flag.sum()} ({fy_nz_flag.mean():.4%})")
    print(f"  Fy range: [{Y_all[:,1].min():.3e}, {Y_all[:,1].max():.3e}]")
    print(f"  Fy mean: {Y_all[:,1].mean():.3e}, std: {Y_all[:,1].std():.3e}")
    if fy_nz_flag.sum()>0:
        nz_vals = Y_all[fy_nz_flag==1,1]
        print(f"  Non-zero Fy range: [{nz_vals.min():.3e}, {nz_vals.max():.3e}]")
        print(f"  Non-zero Fy mean: {nz_vals.mean():.3e}, std: {nz_vals.std():.3e}")

    # ---------- 构建可训练子集（保留全部非零 Fy，再抽零） ----------
    idx_nz = np.where(fy_nz_flag==1)[0]
    idx_z  = np.where(fy_nz_flag==0)[0]

    keep_idx = idx_nz.tolist()
    # 目标：训练集中非零 Fy 占比 ~ 0.20（与 batch 目标不同，batch 由采样器控制）
    target_train_nz_ratio = 0.20
    target_total = len(idx_nz) / max(1e-6, target_train_nz_ratio)
    target_total = int(max(target_total, len(idx_nz) + 1000))

    if MAX_TRAIN_SAMPLES and target_total > MAX_TRAIN_SAMPLES:
        target_total = MAX_TRAIN_SAMPLES

    need_zero = max(0, target_total - len(idx_nz))
    if need_zero > len(idx_z):
        need_zero = len(idx_z)

    rng = np.random.default_rng(SEED)
    sampled_zero = rng.choice(idx_z, size=need_zero, replace=False)
    keep_idx += sampled_zero.tolist()
    keep_idx = np.array(keep_idx, dtype=np.int64)
    rng.shuffle(keep_idx)

    X_sub = X_all[keep_idx]
    Y_sub = Y_all[keep_idx]
    fy_nz_sub = (np.abs(Y_sub[:,1]) > NZ_THRESHOLD_ABS).astype(np.int32)

    print(f"[FY BALANCE] Kept: {len(keep_idx):,} | Non-zero Fy: {fy_nz_sub.sum():,} "
          f"({fy_nz_sub.mean():.3%})")

    # ---------- 分层切分（保证 Val 中也有非零 Fy） ----------
    Xtr, Xva, Ytr, Yva = train_test_split(
        X_sub, Y_sub, test_size=0.20, random_state=SEED, stratify=fy_nz_sub
    )
    print(f"[SPLIT] Train: {len(Xtr):,}, Val: {len(Xva):,}")
    print(f"        FY nz ratio train={ (np.abs(Ytr[:,1])>NZ_THRESHOLD_ABS).mean():.4f}, "
          f"val={ (np.abs(Yva[:,1])>NZ_THRESHOLD_ABS).mean():.4f}")

    # ---------- 特征标准化（仅用训练集统计） ----------
    mu, std = zscore_fit(Xtr)
    Xtr = zscore_apply(Xtr, mu, std).astype(np.float32)
    Xva = zscore_apply(Xva, mu, std).astype(np.float32)

    # ---------- 数据集与 DataLoader ----------
    train_ds = ForceDataset(Xtr, Ytr)
    val_ds   = ForceDataset(Xva, Yva)

    # 训练集采样权重：提高 Fy 非零样本被采到的概率
    fy_nz_tr = (np.abs(Ytr[:,1]) > NZ_THRESHOLD_ABS).astype(np.int64)
    # 解一个简单权重，使得采样后非零比例逼近 TARGET_FY_NZ_RATIO_IN_BATCH
    # 令零类权重=1，非零类权重=k，使得期望非零占比 = k*p / (k*p + (1-p))
    p = fy_nz_tr.mean() + 1e-9
    target = TARGET_FY_NZ_RATIO_IN_BATCH
    k = (target * (1 - p)) / (p * (1 - target))  # 推导自上式
    k = float(np.clip(k, 1.0, 50.0))
    w = np.where(fy_nz_tr==1, k, 1.0).astype(np.float64)

    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

    # ---------- 模型 ----------
    model = GRU_FY_TwoHead(in_dim=len(FEATURES), hidden=256, layers=1, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                       patience=5, verbose=False)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0, device=device))  # 正类很稀少

    # 目标分量标准化（简单做：每分量 z-score，防止尺度差引起不稳）
    y_mean = Ytr.mean(axis=0, keepdims=True)
    y_std  = Ytr.std(axis=0, keepdims=True) + 1e-6
    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t  = torch.tensor(y_std,  dtype=torch.float32, device=device)

    def loss_fn(vec_pred, fy_logits, y_true, epoch):
        # Fx/Fz 回归（标准化后）
        y_std_pred = (vec_pred - y_mean_t) / y_std_t
        y_std_true = (y_true   - y_mean_t) / y_std_t
        loss_fx = F.smooth_l1_loss(y_std_pred[:,0], y_std_true[:,0])
        loss_fz = F.smooth_l1_loss(y_std_pred[:,2], y_std_true[:,2])

        # Fy 分类：放宽阈值构造标签，增大正类（比如 0.33）
        cls_th = max(0.25, 0.5 * NZ_THRESHOLD_ABS)
        fy_true = y_true[:,1]
        fy_cls_y = (torch.abs(fy_true) > cls_th).float()
        loss_fy_cls = bce(fy_logits, fy_cls_y)

        # Fy 回归：只对真·非零(评估阈值)样本强制拟合
        nz_mask = (torch.abs(fy_true) > NZ_THRESHOLD_ABS)
        if nz_mask.any():
            loss_fy_reg = F.smooth_l1_loss(vec_pred[nz_mask,1], fy_true[nz_mask])
            # 防“全零化”的幅度下限：鼓励 |pred| 不低于 cls_th
            margin = F.relu(cls_th - torch.abs(vec_pred[nz_mask,1])).mean()
        else:
            loss_fy_reg = vec_pred.new_tensor(0.0)
            margin = vec_pred.new_tensor(0.0)

        # 向量模一致性（轻权重）
        mag_true = torch.linalg.vector_norm(y_true, dim=1)
        mag_pred = torch.linalg.vector_norm(vec_pred, dim=1)
        mag_loss = F.smooth_l1_loss(mag_pred, mag_true)

        alpha_reg = 0.3 if epoch < 10 else 0.7
        total = (loss_fx + loss_fz
                 + 1.0 * loss_fy_cls
                 + alpha_reg * loss_fy_reg
                 + 0.2 * margin
                 + 0.2 * mag_loss)
        return torch.nan_to_num(total, 0.0, 1e6, 1e6)

    # ---------- 训练 ----------
    best_val = float('inf')
    best_state = None
    no_improve = 0
    print(f"[TRAIN] EPOCHS={EPOCHS}, BATCH={BATCH_SIZE}, target_batch_FyNZ≈{TARGET_FY_NZ_RATIO_IN_BATCH:.2f}")
    print(f"[LOSS] NZ_THRESHOLD_ABS={NZ_THRESHOLD_ABS:.3f}, relaxed_cls_th≈{max(0.25, 0.5*NZ_THRESHOLD_ABS):.3f}")

    for ep in range(1, EPOCHS+1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0; nsum = 0
        # 监控 batch 内 Fy 非零占比（真/预测）
        run_true_nz = []; run_pred_nz = []

        for xb, yb in train_dl:
            xb = xb.to(device); yb = yb.to(device)
            vp, fy_logits = model(xb)
            loss = loss_fn(vp, fy_logits, yb, ep)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()

            loss_sum += float(loss.item()) * len(xb)
            nsum += len(xb)

            with torch.no_grad():
                # 统计 batch 里的 Fy 非零比例（真/按评估阈值的“回归预测”）
                run_true_nz.append((torch.abs(yb[:,1]) > NZ_THRESHOLD_ABS).float().mean().item())
                run_pred_nz.append((torch.abs(vp[:,1]) > NZ_THRESHOLD_ABS).float().mean().item())

        train_loss = loss_sum / max(1, nsum)

        # 验证
        model.eval()
        with torch.no_grad():
            vloss_sum = 0.0; vsum = 0
            for xb, yb in val_dl:
                xb = xb.to(device); yb = yb.to(device)
                vp, fy_logits = model(xb)
                vloss = loss_fn(vp, fy_logits, yb, ep)
                vloss_sum += float(vloss.item()) * len(xb)
                vsum += len(xb)
            val_loss = vloss_sum / max(1, vsum)

        Fx_r2, Fy_r2, Fz_r2, r2_all, r2_nz, nz_true, nz_pred = evaluate(model, val_dl, device, NZ_THRESHOLD_ABS)

        print(f"[Epoch {ep:03d}/{EPOCHS}] "
              f"Loss: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"R²: Fx({Fx_r2:.3f}) Fy({Fy_r2:.3f}) Fz({Fz_r2:.3f}) | "
              f"|F| R²(all/nz)=({r2_all:.3f}/{r2_nz:.3f}) | "
              f"FY NZ(train mean ~{np.mean(run_true_nz):.3f}/{np.mean(run_pred_nz):.3f}) "
              f"val {nz_true:.3f}/{nz_pred:.3f} | "
              f"time={time.time()-t0:.1f}s")

        sched.step(val_loss)

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[STOP] Early stopping at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------- 最终评估 ----------
    print("\n[FINAL EVALUATION]")
    Fx_r2, Fy_r2, Fz_r2, r2_all, r2_nz, nz_true, nz_pred = evaluate(model, val_dl, device, NZ_THRESHOLD_ABS)
    print(f"R² Scores - Fx: {Fx_r2:.4f}, Fy: {Fy_r2:.4f}, Fz: {Fz_r2:.4f}")
    print(f"|F| R² (all): {r2_all:.4f}   |F| R² (nz): {r2_nz:.4f}")
    print(f"FY non-zero (val): true={nz_true:.4f}, pred={nz_pred:.4f}")

    plot_val_scatters(model, val_dl, device)
    plot_val_scatters_pretty(model, val_dl, device, outdir="./output/scatter_val", max_points=4000)

    print("\n[DONE]")

if __name__ == "__main__":
    seed_everything()
    main()
