# -*- coding: utf-8 -*-
"""
interpret_addons.py
把解释/诊断方法独立成库：GAM (pyGAM)、PLSRegression（监督式因子）、Ridge 多输出线性代理
不依赖主程序内部实现细节；仅依赖传入的 DataFrame、列名、输出目录 Path。
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 确保无界面环境能画图
import matplotlib.pyplot as plt


def run_gam(trZ: pd.DataFrame,
            tr_df: pd.DataFrame,
            FEATURES: list,
            TARGET_COLS: list,
            out_dir: Path,
            gam_splines: int = 10,
            gam_lam: float = 0.3,
            gam_max_samples: int = 200_000) -> None:
    """
    用 pyGAM 对 (Fx,Fy,Fz) 训练“每特征一条样条”的加性模型，导出 PDP 曲线与 summary。
    - trZ: 含标准化后的训练特征（DataFrame）
    - tr_df: 原训练 DataFrame（拿物理单位真值）
    - FEATURES: 特征名列表
    - TARGET_COLS: 目标列名 ["Fx","Fy","Fz"]
    - out_dir: 本次运行的输出目录 Path（如 artifacts_vec/GRU_时间戳）
    """
    try:
        from pygam import LinearGAM, s
    except Exception as e:
        print("[GAM] Skipped (pygam not available):", e)
        return

    gam_dir = Path(out_dir) / "gam"
    gam_dir.mkdir(parents=True, exist_ok=True)
    print("[GAM] Training pyGAM interpretable models for Fx/Fy/Fz ...")

    # 使用训练集：标准化特征 + 物理单位真值
    Z_tr = trZ[FEATURES].to_numpy(dtype=np.float32)
    Y_tr = tr_df[TARGET_COLS].to_numpy(dtype=np.float32)

    # 防止样本过大
    if len(Z_tr) > gam_max_samples:
        idx = np.random.choice(len(Z_tr), gam_max_samples, replace=False)
        Z_tr = Z_tr[idx]
        Y_tr = Y_tr[idx]

    p = Z_tr.shape[1]
    if p == 0:
        print("[GAM] No features found, skip.")
        return

    for j, name in enumerate(["Fx", "Fy", "Fz"]):
        y = Y_tr[:, j]

        # 正确叠加样条项：从 s(0) 起
        terms = s(0, n_splines=gam_splines)
        for i in range(1, p):
            terms = terms + s(i, n_splines=gam_splines)

        gam = LinearGAM(terms, lam=gam_lam).fit(Z_tr, y)

        # 每个特征的 1D PDP
        for k, feat in enumerate(FEATURES):
            XX = gam.generate_X_grid(term=k)
            pdep = gam.partial_dependence(term=k, X=XX)

            # CSV
            pd.DataFrame({
                "feature": feat,
                "x": XX[:, k],
                f"PDP_{name}": pdep
            }).to_csv(gam_dir / f"pdp_{name}_{feat}.csv", index=False)

            # PNG
            plt.figure(figsize=(5, 3.2))
            plt.plot(XX[:, k], pdep, lw=2)
            plt.title(f"GAM PDP: {name} vs {feat}")
            plt.xlabel(feat)
            plt.ylabel(f"{name} (effect)")
            plt.tight_layout()
            plt.savefig(gam_dir / f"pdp_{name}_{feat}.png", dpi=150)
            plt.close()

        # 保存 summary（注意 pyGAM 的 p-value 警告）
        try:
            with open(gam_dir / f"summary_{name}.txt", "w", encoding="utf-8") as f:
                f.write(str(gam.summary()))
        except Exception as e:
            warnings.warn(f"[GAM] summary write failed for {name}: {e}")

    print("[GAM] Done →", gam_dir)


def run_pls(trZ: pd.DataFrame,
            tr_df: pd.DataFrame,
            FEATURES: list,
            TARGET_COLS: list,
            out_dir: Path,
            n_comp: int | None = None,
            topk: int = 15) -> None:
    """
    PLSRegression（监督式因子），导出 x/y 权重和 Top-K 特征条形图。
    """
    try:
        from sklearn.cross_decomposition import PLSRegression
    except Exception as e:
        print("[PLS] Skipped (scikit-learn not available):", e)
        return

    pls_dir = Path(out_dir) / "interpret_pls"
    pls_dir.mkdir(parents=True, exist_ok=True)

    Z_tr = trZ[FEATURES].to_numpy(dtype=np.float32)
    Y_tr = tr_df[TARGET_COLS].to_numpy(dtype=np.float32)

    if n_comp is None:
        n_comp = int(min(3, max(1, len(FEATURES))))

    pls = PLSRegression(n_components=n_comp).fit(Z_tr, Y_tr)

    xw = pd.DataFrame(pls.x_weights_, columns=[f"PLS{i+1}" for i in range(n_comp)], index=FEATURES)
    yw = pd.DataFrame(pls.y_weights_, columns=[f"PLS{i+1}" for i in range(n_comp)], index=["Fx","Fy","Fz"])
    xw.to_csv(pls_dir/"pls_x_weights.csv")
    yw.to_csv(pls_dir/"pls_y_weights.csv")

    # Top-K 可视化
    for i in range(n_comp):
        w = xw.iloc[:, i].abs().sort_values(ascending=False)
        k = min(topk, len(w))
        plt.figure(figsize=(7, 0.4*k+1))
        plt.barh(w.index[:k][::-1], w.values[:k][::-1])
        plt.title(f"PLS{i+1} | Top feature weights (abs)")
        plt.tight_layout()
        plt.savefig(pls_dir/f"pls{i+1}_top_features.png", dpi=150)
        plt.close()

    # 训练集 R²（粗参考）
    Y_hat = pls.predict(Z_tr)
    def _r2(a,b):
        a=a.reshape(-1); b=b.reshape(-1)
        return 1 - ((a-b)**2).sum() / (((a-a.mean())**2).sum() + 1e-12)
    r2_fx = _r2(Y_tr[:,0], Y_hat[:,0])
    r2_fy = _r2(Y_tr[:,1], Y_hat[:,1])
    r2_fz = _r2(Y_tr[:,2], Y_hat[:,2])
    with open(pls_dir/"pls_train_r2.txt","w",encoding="utf-8") as f:
        f.write(f"R2_train: Fx={r2_fx:.4f}, Fy={r2_fy:.4f}, Fz={r2_fz:.4f}\n")

    print("[PLS] Done →", pls_dir)


def run_linear_ridge(trZ: pd.DataFrame,
                     tr_df: pd.DataFrame,
                     FEATURES: list,
                     TARGET_COLS: list,
                     out_dir: Path,
                     alphas: np.ndarray | None = None,
                     topk: int = 15) -> None:
    """
    Ridge 多输出线性代理：导出系数矩阵与热力图、每个输出的 Top-K。
    """
    try:
        from sklearn.linear_model import RidgeCV
        from sklearn.multioutput import MultiOutputRegressor
    except Exception as e:
        print("[LIN] Skipped (scikit-learn not available):", e)
        return

    lin_dir = Path(out_dir) / "interpret_linear"
    lin_dir.mkdir(parents=True, exist_ok=True)

    Z_tr = trZ[FEATURES].to_numpy(dtype=np.float32)
    Y_tr = tr_df[TARGET_COLS].to_numpy(dtype=np.float32)

    if alphas is None:
        alphas = np.logspace(-4, 3, 12)

    ridge = MultiOutputRegressor(RidgeCV(alphas=alphas, cv=5)).fit(Z_tr, Y_tr)

    # 叠出 [p,3] 系数矩阵
    W = np.vstack([est.coef_ for est in ridge.estimators_]).T
    dfW = pd.DataFrame(W, index=FEATURES, columns=["Fx","Fy","Fz"])
    dfW.to_csv(lin_dir/"lin_coef.csv")

    # 各输出 Top-K
    for col in ["Fx","Fy","Fz"]:
        wabs = dfW[col].abs().sort_values(ascending=False).head(min(topk, len(dfW)))
        wabs.to_csv(lin_dir/f"lin_top_{col}.csv", header=[f"|coef|_{col}"])

    # 热力图
    plt.figure(figsize=(max(6, 0.5*len(FEATURES)), 5))
    plt.imshow(np.abs(dfW.values), aspect="auto")
    plt.xticks(ticks=range(3), labels=["Fx","Fy","Fz"])
    plt.yticks(ticks=range(len(FEATURES)), labels=FEATURES)
    plt.colorbar(label="|coef|")
    plt.title("Linear surrogate (Ridge) | coefficients (abs)")
    plt.tight_layout()
    plt.savefig(lin_dir/"lin_coef.png", dpi=150)
    plt.close()

    print("[LIN] Done →", lin_dir)
