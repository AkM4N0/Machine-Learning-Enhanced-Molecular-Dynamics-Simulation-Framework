import os
import re
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg')  # 或 'Agg'、'Qt5Agg'

# === 配置 ===
input_cols = [
    'd', 'r0', 'h0', 'r0_over_h0',
    'pair_0_1_qrel_x', 'pair_0_1_qrel_y', 'pair_0_1_qrel_z', 'pair_0_1_qrel_w',
]
target_cols = ['U_int_Fx', 'U_int_Fy', 'U_int_Fz']
seq_len = 5
n_epochs = 30
batch_size = 32

# 方案1：分量加权（可调）。先用 (1,1,1)；若想更重视 Fz，可试 (1,1,2~5)
COMP_WEIGHTS = (1.0, 1.0, 1.0)
HUBER_DELTA = 1.0  # 可按需调大/调小

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ======================================================================
# 模型：MLP（接收序列窗口，内部展平，多输出=3）
# ======================================================================
class MLPRegressor(nn.Module):
    def __init__(self, input_size, seq_len, hidden_sizes=(512, 256, 128), dropout=0.2, out_dim=3):
        super().__init__()
        in_feats = input_size * seq_len
        layers, last = [], in_feats
        for hs in hidden_sizes:
            layers += [nn.Linear(last, hs), nn.ReLU(), nn.Dropout(dropout)]
            last = hs
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: [B, T, F]
        x = x.reshape(x.size(0), -1)  # [B, T*F]
        return self.net(x)            # [B, 3]

# === 序列构建 ===
def create_sequences(X, Y, seq_len):
    X_seq, Y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        Y_seq.append(Y[i + seq_len])
    return np.asarray(X_seq), np.asarray(Y_seq)

def sort_key(name: str):
    m = re.search(r'_(\d+)_(\d+)\.csv$', name)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    nums = re.findall(r'\d+', name)
    return tuple(map(int, nums)) if nums else (float('inf'), float('inf'))

def _create_seq_from_df(df, scaler_X, scaler_Y, input_cols, target_cols, seq_len):
    X = scaler_X.transform(df[input_cols])
    Y = scaler_Y.transform(df[target_cols])
    X_seq, Y_seq = create_sequences(X, Y, seq_len)
    return X_seq, Y_seq

def make_loaders_agg_weighted(
    all_csv_files, input_cols, target_cols, seq_len,
    test_size=0.2, random_state=42,
    q_low=0.1, weight_hi=10.0, weight_lo=1.0,
    batch_size=32
):
    # 按文件分组切分
    n_files = len(all_csv_files)
    file_idx = np.arange(n_files)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_idx, val_idx = next(gss.split(file_idx, groups=file_idx))
    train_files = [all_csv_files[i] for i in tr_idx]
    val_files   = [all_csv_files[i] for i in val_idx]

    # 仅用训练文件拟合 scaler（避免泄漏）
    dfX_train_rows, dfY_train_rows = [], []
    for p in train_files:
        df = pd.read_csv(p).fillna(0)
        dfX_train_rows.append(df[input_cols])
        dfY_train_rows.append(df[target_cols])
    dfX_train_rows = pd.concat(dfX_train_rows, axis=0, ignore_index=True)
    dfY_train_rows = pd.concat(dfY_train_rows, axis=0, ignore_index=True)
    scaler_X = RobustScaler().fit(dfX_train_rows)
    scaler_Y = RobustScaler().fit(dfY_train_rows)

    # 造序列并汇总
    X_tr_list, Y_tr_list = [], []
    for p in train_files:
        df = pd.read_csv(p).fillna(0)
        Xi, Yi = _create_seq_from_df(df, scaler_X, scaler_Y, input_cols, target_cols, seq_len)
        if len(Xi): X_tr_list.append(Xi); Y_tr_list.append(Yi)
    X_va_list, Y_va_list = [], []
    for p in val_files:
        df = pd.read_csv(p).fillna(0)
        Xi, Yi = _create_seq_from_df(df, scaler_X, scaler_Y, input_cols, target_cols, seq_len)
        if len(Xi): X_va_list.append(Xi); Y_va_list.append(Yi)

    X_train = np.concatenate(X_tr_list, axis=0) if X_tr_list else np.zeros((0, seq_len, len(input_cols)))
    Y_train = np.concatenate(Y_tr_list, axis=0) if Y_tr_list else np.zeros((0, 3))
    X_val   = np.concatenate(X_va_list, axis=0) if X_va_list else np.zeros((0, seq_len, len(input_cols)))
    Y_val   = np.concatenate(Y_va_list, axis=0) if Y_va_list else np.zeros((0, 3))

    # 基于 |F|（物理域）计算采样权重，降权近零样本
    Y_train_phys = scaler_Y.inverse_transform(Y_train)
    Fnorm = np.linalg.norm(Y_train_phys, axis=1)
    thr = np.quantile(Fnorm, q_low)
    sample_weights = np.where(Fnorm < thr, weight_lo, weight_hi).astype(np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(Y_train, dtype=torch.float32))
    val_ds   = TensorDataset(torch.tensor(X_val,   dtype=torch.float32),
                             torch.tensor(Y_val,   dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    meta = {
        "train_files": train_files,
        "val_files": val_files,
        "thr": float(thr),
        "q_low": q_low,
        "weight_hi": weight_hi,
        "weight_lo": weight_lo,
    }
    return train_loader, val_loader, scaler_X, scaler_Y, meta

def plot_saliency_heatmap(model, X_seq, input_cols, save_path="saliency_heatmap.png", sample_limit=50):
    model.train()
    X_subset = np.asarray(X_seq[:sample_limit])
    X_tensor = torch.tensor(X_subset, dtype=torch.float32, requires_grad=True, device=device)
    y_vec = model(X_tensor)                  # [N,3]
    y_mag = torch.norm(y_vec, dim=1)         # [N]
    y_mag[0].backward()
    sal = X_tensor.grad.abs().detach().cpu().numpy()  # [N,T,F]
    mean_sal = np.mean(sal, axis=0)                  # [T,F]
    plt.figure(figsize=(10, 6))
    plt.imshow(mean_sal, cmap="viridis", aspect="auto")
    plt.colorbar(label="Saliency")
    plt.title("Saliency Heatmap (Time × Feature)")
    plt.xlabel("Feature Index"); plt.ylabel("Time Step")
    plt.xticks(ticks=np.arange(len(input_cols)), labels=input_cols, rotation=45)
    plt.tight_layout(); plt.savefig(save_path)
    print(f"[INFO] Saliency heatmap saved to {save_path}.")

# === 指标
def mae_rmse_r2(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# === 训练（A + C + 方案1：分量加权 Huber）
def train_global(model, train_loader, val_loader, scaler_Y,
                 comp_weights=(1.0,1.0,1.0), huber_delta=1.0,
                 n_epochs=30, patience=5, tag="ALL"):
    best_val, best_state, epochs_no_improve = float('inf'), None, 0

    # HuberLoss(reduction='none') 返回逐元素损失，便于做分量加权
    huber = nn.HuberLoss(reduction='none', delta=huber_delta).to(device)
    comp_w = torch.tensor(comp_weights, device=device).view(1, 3)  # [1,3]

    for epoch in range(n_epochs):
        # ---- train ----
        model.train(); total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)   # [B,T,F], [B,3]
            pred = model(xb)                         # [B,3]
            loss_raw = huber(pred, yb)               # [B,3]
            loss = (loss_raw * comp_w).mean()        # 分量加权再取均值
            # 反向
            for p in model.parameters():  # 避免残留梯度
                if p.grad is not None:
                    p.grad.detach_(); p.grad.zero_()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            # 优化器放在外面方便替换
            optimizer.step()
            total += loss.item()
        train_loss = total / max(1, len(train_loader))

        # ---- val ----
        model.eval(); val_loss = 0.0
        Yt_list, Yp_list = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                lraw = huber(pred, yb)               # [B,3]
                vloss = (lraw * comp_w).mean()
                val_loss += vloss.item()
                Yt_list.append(yb.cpu().numpy())
                Yp_list.append(pred.cpu().numpy())
        val_loss /= max(1, len(val_loader))
        Y_true_std = np.concatenate(Yt_list, axis=0)
        Y_pred_std = np.concatenate(Yp_list, axis=0)

        # 反缩放回物理域做指标
        y_true_inv = scaler_Y.inverse_transform(Y_true_std)
        y_pred_inv = scaler_Y.inverse_transform(Y_pred_std)

        def _metrics():
            mFx = mae_rmse_r2(y_true_inv[:, 0], y_pred_inv[:, 0])
            mFy = mae_rmse_r2(y_true_inv[:, 1], y_pred_inv[:, 1])
            mFz = mae_rmse_r2(y_true_inv[:, 2], y_pred_inv[:, 2])
            mag_t, mag_p = np.linalg.norm(y_true_inv, axis=1), np.linalg.norm(y_pred_inv, axis=1)
            mMag = mae_rmse_r2(mag_t, mag_p)
            return mFx, mFy, mFz, mMag

        mFx, mFy, mFz, mMag = _metrics()
        print(f"[{tag}][Ep{epoch+1:03d}] Train={train_loss:.4f} | Val={val_loss:.4f} | "
              f"Fx MAE/RMSE/R2={mFx} | Fy={mFy} | Fz={mFz} | |F|={mMag}")

        if val_loss < best_val:
            best_val, best_state, epochs_no_improve = val_loss, model.state_dict(), 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Early Stopping] 连续 {patience} 次无提升，停止。"); break

    # —— 输出三张散点图 ——（最后一轮）
    def scatter_diag(y_true, y_pred, name, fn):
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        plt.figure(figsize=(5,5))
        plt.scatter(y_true, y_pred, s=6, alpha=0.4)
        plt.plot([lo, hi], [lo, hi], 'r--', label='Ideal')
        plt.xlabel(f"True {name}"); plt.ylabel(f"Pred {name}")
        plt.title(f"{name}: Pred vs True"); plt.legend()
        plt.tight_layout(); plt.savefig(fn); plt.close()

    scatter_diag(y_true_inv[:,0], y_pred_inv[:,0], "Fx", f"pred_vs_true_Fx_{tag}.png")
    scatter_diag(y_true_inv[:,1], y_pred_inv[:,1], "Fy", f"pred_vs_true_Fy_{tag}.png")
    scatter_diag(y_true_inv[:,2], y_pred_inv[:,2], "Fz", f"pred_vs_true_Fz_{tag}.png")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def plot_input_response(model, scaler_X, scaler_Y, base_inputs, input_cols, save_dir='input_response'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    for var in input_cols:
        x_vals = np.linspace(base_inputs[var]*0.5, base_inputs[var]*1.5, 100)
        yFx, yFy, yFz, yMag = [], [], [], []
        for v in x_vals:
            sample = base_inputs.copy()
            sample[var] = v
            if 'r0' in sample and 'h0' in sample:
                sample['r0_over_h0'] = (sample['r0'] / sample['h0']) if sample['h0'] != 0 else 0.0
            x_arr = pd.DataFrame([sample])[input_cols]
            x_scaled = scaler_X.transform(x_arr)
            x_rep = np.repeat(x_scaled, seq_len, axis=0)
            x_seq = torch.tensor(x_rep[np.newaxis, ...], dtype=torch.float32, device=device)
            with torch.no_grad():
                pred_std = model(x_seq).cpu().numpy()
            pred = scaler_Y.inverse_transform(pred_std)[0]
            yFx.append(pred[0]); yFy.append(pred[1]); yFz.append(pred[2]); yMag.append(np.linalg.norm(pred))
        for name, arr in zip(["Fx","Fy","Fz","Fmag"], [yFx,yFy,yFz,yMag]):
            plt.figure(); plt.plot(x_vals, arr)
            plt.xlabel(var); plt.ylabel(f"Predicted {name}")
            plt.title(f"{name} vs {var}"); plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{name}_vs_{var}.png")); plt.close()

def plot_PCA(X_seq, y_seq):
    X_seq_2D = X_seq.reshape((X_seq.shape[0], -1))
    pca = PCA(n_components=7)
    X_pca = pca.fit_transform(X_seq_2D)
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=np.linalg.norm(y_seq, axis=1), cmap='viridis', alpha=0.6)
    plt.colorbar(label='|F|'); plt.title("PCA Projection of Input Sequences")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = Path(__file__).parent / "nn_input_data_xy"
    all_csv_files = sorted(
        DATA_DIR.glob("nn_input_data_*_*.csv"),
        key=lambda p: sort_key(p.name)
    )
    print(f"[INFO] Found {len(all_csv_files)} datasets in {DATA_DIR}")

    # 聚合 + 分组切分 + 采样权重
    train_loader, val_loader, scaler_X, scaler_Y, meta = make_loaders_agg_weighted(
        all_csv_files=all_csv_files,
        input_cols=input_cols,
        target_cols=target_cols,
        seq_len=seq_len,
        test_size=0.2,
        random_state=42,
        q_low=0.1,
        weight_hi=10.0,
        weight_lo=1.0,
        batch_size=batch_size
    )
    print(f"[INFO] Train files: {len(meta['train_files'])}, Val files: {len(meta['val_files'])}, "
          f"Fnorm thr={meta['thr']:.4g} (q={meta['q_low']})")

    # 初始化/续训模型与优化器
    model = MLPRegressor(input_size=len(input_cols), seq_len=seq_len, out_dim=3).to(device)
    model_path = os.path.join(os.path.dirname(__file__), "final_model_fxyz.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[INFO] Loaded previous model from: {model_path}")
    else:
        print("[INFO] No existing model found, starting from scratch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 统一训练（方案1 已生效）
    model = train_global(model, train_loader, val_loader, scaler_Y,
                         comp_weights=COMP_WEIGHTS, huber_delta=HUBER_DELTA,
                         n_epochs=n_epochs, patience=5, tag="ALL")

    # 保存
    OUT_DIR = os.path.join(DATA_DIR, "out"); os.makedirs(OUT_DIR, exist_ok=True)
    save_path = os.path.join(OUT_DIR, "final_model_fxyz.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\n[SUCCESS] Force model (Fx,Fy,Fz) saved to {save_path}")

    # 单变量响应曲线（基于训练文件的均值）
    df0 = pd.concat([pd.read_csv(p).fillna(0)[input_cols] for p in meta['train_files']], axis=0)
    base_inputs = df0.mean().to_dict()
    plot_input_response(model, scaler_X, scaler_Y, base_inputs, input_cols)

    # saliency：取一个 batch 做显著性图
    last_Xseq = None
    for xb, _ in train_loader:
        last_Xseq = xb.numpy()
        break
    print("[INFO] Saliency...")
    if last_Xseq is not None and len(last_Xseq) > 0:
        plot_saliency_heatmap(model, last_Xseq, input_cols)
