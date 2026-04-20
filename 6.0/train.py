# Rewriting the user's code to directly use `merged_df` instead of dynamic loading

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# Use the previously merged dataframe
df = pd.read_csv("merged_nn_input_data.csv")
root_dir = "."

# Create output directory
def make_output_dir(root_dir=".", subfolder="ml_output"):
    output_path = os.path.join(root_dir, subfolder)
    os.makedirs(output_path, exist_ok=True)
    print(f"[INFO] Output directory created at: {output_path}")
    return output_path

# Features and target
input_cols = ['d', 'r0', 'h0', 'r0_over_h0',
              'pair_0_1_angle1', 'pair_0_1_angle2', 'pair_0_1_angle3']
target_col = 'target_energy'

# Standardization
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(df[input_cols])
y_scaled = scaler_y.fit_transform(df[[target_col]])

# Create sequences
def create_sequences(X, y, seq_len=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_len=5)

# Split
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define model
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

model = LSTMRegressor(input_size=len(input_cols))
criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Train loop
train_losses, val_losses = [], []
best_val_loss = float('inf')
patience = 10
epochs_no_improve = 0
best_model_state = None

for epoch in range(100):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb).squeeze()
        loss = criterion(pred, yb.squeeze())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    preds, truths = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb).squeeze()
            loss = criterion(pred, yb.squeeze())
            val_loss += loss.item()
            preds.append(pred.cpu().numpy())
            truths.append(yb.squeeze().cpu().numpy())
    val_loss /= len(val_loader)

    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # === 每个 epoch 输出日志 ===
    print(f"Epoch {epoch + 1:03d}: "
          f"Train Loss = {train_loss:.6f}, "
          f"Val Loss = {val_loss:.6f}")

    # 记录最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"[Early Stopping] No improvement for {patience} epochs.")
            break

# Load best model
model.load_state_dict(best_model_state)
output_dir = make_output_dir(root_dir)
torch.save(best_model_state, os.path.join(output_dir, "lstm_best_model.pt"))

# Evaluation
y_true = np.concatenate(truths)
y_pred = np.concatenate(preds)
y_true_inv = scaler_y.inverse_transform(y_true.reshape(-1, 1))
y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1))

mae = mean_absolute_error(y_true_inv, y_pred_inv)
rmse = mean_squared_error(y_true_inv, y_pred_inv, squared=False)
r2 = r2_score(y_true_inv, y_pred_inv)

# Plotting & saving
def visualize_training(train_losses, val_losses, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Huber)")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lstm_loss_curve.png"))
    plt.close()

def visualize_predictions(y_true, y_pred, output_dir):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True Energy")
    plt.ylabel("Predicted Energy")
    plt.title("Predicted vs True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lstm_pred_vs_true.png"))
    plt.close()

def visualize_error_distribution(y_true, y_pred, output_dir):
    errors = y_pred - y_true
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Prediction Error Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lstm_error_hist.png"))
    plt.close()

def save_data_arrays(y_true, y_pred, train_losses, val_losses, output_dir):
    pd.DataFrame({"True": y_true.flatten(), "Predicted": y_pred.flatten()}).to_csv(
        os.path.join(output_dir, "predictions.csv"), index=False)
    pd.DataFrame({"Train Loss": train_losses, "Val Loss": val_losses}).to_csv(
        os.path.join(output_dir, "loss_curve.csv"), index=False)

# Save all outputs
visualize_training(train_losses, val_losses, output_dir)
visualize_predictions(y_true_inv, y_pred_inv, output_dir)
visualize_error_distribution(y_true_inv, y_pred_inv, output_dir)
save_data_arrays(y_true_inv, y_pred_inv, train_losses, val_losses, output_dir)

# Final metrics
mae, rmse, r2
device = torch.device("cpu")  # 或 "cuda" if torch.cuda.is_available()

# === 推理：计算能量 + 自动微分得到每个时间步的力 ===
force_records = []
sample_inputs, _ = next(iter(val_loader))
sample_inputs = sample_inputs.to(device)

for i in range(min(len(sample_inputs), 32)):  # 前 32 个样本
    X = sample_inputs[i:i+1].clone().detach().requires_grad_(True)
    energy = model(X).squeeze()
    grads = torch.autograd.grad(energy, X, create_graph=False)[0]

    d_forces = -grads[:, :, 0].squeeze().cpu().numpy()  # shape: (seq_len,)
    record = {"pred_energy": energy.item()}
    for t, force in enumerate(d_forces):
        record[f"d_force_t{t}"] = force
    force_records.append(record)

pd.DataFrame(force_records).to_csv("ml_output/energy_scalar_force.csv", index=False)

