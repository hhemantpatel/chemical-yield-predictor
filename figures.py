"""Make the README figures from the data and the models saved by train.py."""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from yieldpred.data import DOMAIN, FEATURES, ROOT, TARGET, YIELD_CAP, load

FIG_DIR = ROOT / "images"
SEED = 42  # must match train.py so the hold-out set is the same


def main():
    df = load()
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    models = {n: joblib.load(ROOT / "models" / r["file"]) for n, r in metrics["models"].items()}
    X, y = df[FEATURES].to_numpy(), df[TARGET].to_numpy()
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    FIG_DIR.mkdir(exist_ok=True)

    # 1. Yield distribution: the pile-up at the cap is censoring.
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(y, bins=60, color="#4C72B0")
    ax.set_xlabel("Reaction yield (%)")
    ax.set_ylabel("Experiments")
    ax.set_title(f"{np.mean(y >= YIELD_CAP):.0%} of runs sit exactly at {YIELD_CAP:.0f} % "
                 "(complete conversion)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "yield_distribution.png", dpi=150)

    # 2. Parity plots: OLS vs Tobit on the hold-out set.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for ax, name in zip(axes, ["Linear (OLS)", "Tobit (censored linear)"]):
        pred = models[name].predict(X_te)
        r = metrics["models"][name]
        ax.scatter(y_te, pred, s=4, alpha=0.35)
        ax.plot([20, 100], [20, 100], "k--", lw=1)
        ax.set_title(f"{name}\nRMSE {r['test_rmse']:.2f} %, R² {r['test_r2']:.3f}")
        ax.set_xlabel("Measured yield (%)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Predicted yield (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "parity_ols_vs_tobit.png", dpi=150)

    # 3. Operating window at mid pressure.
    tobit = models["Tobit (censored linear)"]
    Ts = np.linspace(*DOMAIN["temperature"], 200)
    Cs = np.linspace(*DOMAIN["catalyst_concentration"], 120)
    TT, CC = np.meshgrid(Ts, Cs)
    P = 5.0
    Z = tobit.predict(np.c_[TT.ravel(), np.full(TT.size, P), CC.ravel()]).reshape(TT.shape)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    cf = ax.contourf(TT, CC, Z, levels=np.arange(20, 101, 5), cmap="viridis")
    cs = ax.contour(TT, CC, Z, levels=[90, 95, 99], colors="white", linewidths=1.5)
    ax.clabel(cs, fmt="%d %%")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Catalyst concentration (mol/L)")
    ax.set_title(f"Expected yield at {P:.0f} atm (Tobit model)")
    fig.colorbar(cf, label="Yield (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "operating_window.png", dpi=150)
    print(f"figures written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
