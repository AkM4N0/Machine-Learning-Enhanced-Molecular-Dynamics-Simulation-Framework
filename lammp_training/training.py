import os
import re
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg')  # 或 'Agg'、'Qt5Agg'


# === 配置 ===
input_cols = ['d', 'r0', 'h0', 'r0_over_h0',
              'pair_0_1_angle1', 'pair_0_1_angle2', 'pair_0_1_angle3']
target_col = 'F_interaction'
seq_len = 5
n_epochs = 30
batch_size = 32

# === 使用 GPU（如果可用） ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# === 模型定义 ===
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

# === 序列构建函数 ===
def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq), np.array(y_seq)

def sort_key(fname):
    match = re.search(r"(\d+)", fname)
    return int(match.group()) if match else -1

def plot_saliency_heatmap(model, X_seq, input_cols, save_path="saliency_heatmap.png", sample_limit=50):
    model.train()  # 🔥 保证支持 RNN 反向传播

    X_subset = X_seq[:sample_limit]
    X_subset = np.array(X_subset)  # 保证为规整 numpy 数组
    X_tensor = torch.tensor(X_subset, dtype=torch.float32, requires_grad=True)  # 不要直接 .to(device)
    X_tensor = X_tensor.to(device)
    X_tensor.retain_grad()  # 显式保留 grad

    # 前向预测
    y = model(X_tensor).squeeze()

    # 仅取第一个输出的梯度做热图（避免 batch 多时崩溃）
    y[0].backward()

    # 获取 saliency
    saliency = X_tensor.grad.abs().cpu().numpy()  # [N, T, F]
    mean_saliency = np.mean(saliency, axis=0)  # [T, F]

    # === 画热力图 ===
    plt.figure(figsize=(10, 6))
    plt.imshow(mean_saliency, cmap="viridis", aspect="auto")
    plt.colorbar(label="Saliency")
    plt.title("Saliency Heatmap (Time × Feature)")
    plt.xlabel("Feature Index")
    plt.ylabel("Time Step")
    plt.xticks(ticks=np.arange(len(input_cols)), labels=input_cols, rotation=45)
    plt.tight_layout()
    plt.savefig("saliency_heatmap.png")
    print("[INFO] Saliency heatmap saved.")


# === 训练函数 ===
def train_on_dataset(df, model, scaler_X, scaler_y, criterion, optimizer, patience=5):

    df = df[input_cols + [target_col]].copy()
    df.fillna(0, inplace=True)

    x_scaled = scaler_X.transform(df[input_cols])
    local_scaler_y = StandardScaler()
    y_scaled = local_scaler_y.fit_transform(df[[target_col]])
    X_seq, y_seq = create_sequences(x_scaled, y_scaled, seq_len)

    if len(X_seq) == 0:
        print("[WARNING] 数据不足，跳过该文件。")
        return model

    # 划分训练和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                            torch.tensor(y_train, dtype=torch.float32)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                          torch.tensor(y_val, dtype=torch.float32)),
                            batch_size=batch_size)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        # === 验证阶段 ===
        model.eval()
        val_loss = 0
        y_true_all = []
        y_pred_all = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze()
                loss = criterion(pred, yb.squeeze())
                val_loss += loss.item()

                y_true_np = np.atleast_1d(yb.squeeze().cpu().numpy())
                y_pred_np = np.atleast_1d(pred.cpu().numpy())

                y_true_all.append(y_true_np)
                y_pred_all.append(y_pred_np)

        val_loss /= len(val_loader)

        # 拼接为整体向量
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)

        # === 反归一化 ===
        y_true_inv = scaler_y.inverse_transform(y_true_all.reshape(-1, 1)).squeeze()
        y_pred_inv = scaler_y.inverse_transform(y_pred_all.reshape(-1, 1)).squeeze()

        print(f"[Train] Epoch {epoch+1:03d}, Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # === Early Stopping 检查 ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Early Stopping] 停止训练: 连续 {patience} 次无提升。")
                break

    def plot_prediction_comparison(y_true, y_pred):
        csv_num = os.path.splitext(os.path.basename(csv_file))[0].split("nn_input_data")[-1]
        save_path = f"pred_vs_true_{csv_num}.png"
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal')
        plt.xlabel("True F_interaction")
        plt.ylabel("Predicted F_interaction")
        plt.title("Prediction vs True")
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.tight_layout()

    # 恢复最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
        mae_rmse_r2(y_true_inv, y_pred_inv)
        #plot_PCA(X_val, y_true_inv)
        plot_prediction_comparison(y_true_inv, y_pred_inv,)

    return model

def plot_input_response(model, scaler_X, scaler_y, base_inputs, input_cols, save_dir='input_response'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    for var in input_cols:
        x_vals = np.linspace(base_inputs[var]*0.5, base_inputs[var]*1.5, 100)
        y_vals = []
        for v in x_vals:
            sample = base_inputs.copy()
            sample[var] = v
            if 'r0' in sample and 'h0' in sample:
                sample['r0_over_h0'] = sample['r0'] / sample['h0']  # 动态更新 r0_over_h0
            x_arr = pd.DataFrame([sample])[input_cols]
            x_scaled = scaler_X.transform(x_arr)  # shape: (1, input_size)
            x_repeated = np.repeat(x_scaled, seq_len, axis=0)  # (seq_len, input_size)
            x_seq = torch.tensor(x_repeated[np.newaxis, ...], dtype=torch.float32).to(
                device)  # (1, seq_len, input_size)

            with torch.no_grad():
                pred = model(x_seq).cpu().item()
            pred_force = scaler_y.inverse_transform([[pred]])[0, 0]
            y_vals.append(pred_force)
        plt.figure()
        plt.plot(x_vals, y_vals)
        plt.xlabel(var)
        plt.ylabel("Predicted Force")
        plt.title(f"Force vs {var}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"force_vs_{var}.png"))
        plt.close()

def mae_rmse_r2(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")


def plot_PCA(X_seq, y_seq):
    X_seq_2D = X_seq.reshape((X_seq.shape[0], -1))  # Flatten sequence
    pca = PCA(n_components=7)
    X_pca = pca.fit_transform(X_seq_2D)

    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_seq.squeeze(), cmap='viridis', alpha=0.6)
    plt.colorbar(label='F_interaction')
    plt.title("PCA Projection of Input Sequences")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()


# === 主程序入口 ===
if __name__ == "__main__":
    all_csv_files = sorted(glob.glob("nn_input_data*.csv"), key=sort_key)
    print(f"[INFO] Found {len(all_csv_files)} datasets.")

    # 用第一个文件拟合标准化器
    df0 = pd.read_csv(all_csv_files[0])
    scaler_X = RobustScaler().fit(df0[input_cols])
    scaler_y = RobustScaler().fit(df0[[target_col]])

    # === 初始化模型 ===
    model = LSTMRegressor(input_size=len(input_cols)).to(device)

    # === 如果存在之前保存的模型，加载继续训练 ===
    model_path = os.path.join(os.path.dirname(__file__), "final_model_fint.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"[INFO] Loaded previous model from: {model_path}")
    else:
        print("[INFO] No existing model found, starting from scratch.")

    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    X_all = []
    for csv_file in all_csv_files:
        print(f"\n[INFO] Training on: {csv_file}")
        df = pd.read_csv(csv_file)
        df.fillna(0, inplace=True)
        x_scaled = scaler_X.transform(df[input_cols])
        X_seq, _ = create_sequences(x_scaled, df[target_col].values, seq_len)
        X_all.append(X_seq)
        model = train_on_dataset(df, model, scaler_X, scaler_y, criterion, optimizer)

    save_path = os.path.join(os.getcwd(), "final_model_fint.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\n[SUCCESS] Force model saved to {save_path}")

    base_inputs = df0[input_cols].mean().to_dict()
    plot_input_response(model, scaler_X, scaler_y, base_inputs, input_cols)
    print("[INFO] Running SHAP Explanation...")
    plot_saliency_heatmap(model, X_seq, input_cols)



