"""Streamlit app:  streamlit run app.py   (run `python train.py` first)."""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.stats import norm

from yieldpred.data import DOMAIN, ROOT, YIELD_CAP, out_of_domain

MODELS_DIR = ROOT / "models"
TOBIT = "Tobit (censored linear)"
LABELS = {"temperature": "Temperature (°C)", "pressure": "Pressure (atm)",
          "catalyst_concentration": "Catalyst concentration (mol/L)"}

st.set_page_config(page_title="Chemical Yield Predictor", layout="wide")


@st.cache_resource
def load_artifacts():
    metrics_path = ROOT / "reports" / "metrics.json"
    if not metrics_path.exists():
        return None, {}
    metrics = json.loads(metrics_path.read_text())
    models = {name: joblib.load(MODELS_DIR / r["file"])
              for name, r in metrics["models"].items()
              if (MODELS_DIR / r["file"]).exists()}
    return metrics, models


metrics, models = load_artifacts()
if not models:
    st.error("No trained models found. Run `python train.py` first.")
    st.stop()

st.title("Chemical Reaction Yield Predictor")
st.caption("Predicts reaction yield from temperature, pressure and catalyst loading. "
           f"Yield is physically capped at {YIELD_CAP:.0f} % (complete conversion); "
           "the Tobit model treats that cap as censoring instead of as an exact value.")

# -------------------- Sidebar --------------------
with st.sidebar:
    names = list(models)
    model_name = st.selectbox("Model", names, index=names.index(TOBIT) if TOBIT in names else 0)
    r = metrics["models"][model_name]
    st.markdown(f"**Hold-out test:** RMSE {r['test_rmse']:.2f} %, R² {r['test_r2']:.3f}  \n"
                f"**5-fold CV RMSE:** {r['cv_rmse_mean']:.2f} ± {r['cv_rmse_std']:.2f} %")
    st.divider()
    st.markdown("**Operating conditions** (training range)")
    T = st.slider(LABELS["temperature"], *DOMAIN["temperature"], 150.0, 1.0)
    P = st.slider(LABELS["pressure"], *DOMAIN["pressure"], 5.0, 0.1)
    C = st.slider(LABELS["catalyst_concentration"], *DOMAIN["catalyst_concentration"], 0.25, 0.01)

model = models[model_name]
x = np.array([[T, P, C]])
pred = float(model.predict(x)[0])

tab_pred, tab_sens, tab_window, tab_models = st.tabs(
    ["Prediction", "Sensitivity", "Operating window", "Model comparison"])

# -------------------- Prediction --------------------
with tab_pred:
    if out_of_domain(T, P, C):
        st.warning("Outside the training range: " + ", ".join(out_of_domain(T, P, C)))
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected yield", f"{pred:.1f} %")
    if TOBIT in models:
        tobit = models[TOBIT]
        mu = float(tobit[-1].latent_mean(tobit[0].transform(x))[0])
        sigma = tobit[-1].sigma_
        lo, hi = mu - 1.645 * sigma, mu + 1.645 * sigma
        c2.metric("90 % range of a single run",
                  f"{min(lo, YIELD_CAP):.1f} – {min(hi, YIELD_CAP):.1f} %")
        c3.metric("P(complete conversion)", f"{norm.sf((YIELD_CAP - mu) / sigma):.0%}")
        st.caption(f"Range and conversion probability come from the Tobit model "
                   f"(run-to-run noise σ = {sigma:.2f} %), whichever model is selected above.")

# -------------------- Sensitivity --------------------
with tab_sens:
    st.markdown("Each panel varies one input across its training range while the "
                "other two stay at the sidebar values.")
    base = {"temperature": T, "pressure": P, "catalyst_concentration": C}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (feat, (lo, hi)) in zip(axes, DOMAIN.items()):
        grid = np.linspace(lo, hi, 120)
        X = np.tile([base[f] for f in DOMAIN], (len(grid), 1))
        X[:, list(DOMAIN).index(feat)] = grid
        for name in dict.fromkeys([model_name, TOBIT]):
            if name in models:
                ax.plot(grid, models[name].predict(X), label=name,
                        ls="-" if name == model_name else "--")
        ax.axvline(base[feat], color="grey", lw=0.8, ls=":")
        ax.set_xlabel(LABELS[feat])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Predicted yield (%)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)

# -------------------- Operating window --------------------
with tab_window:
    st.markdown("Find the **mildest conditions** that reach a target yield: the lowest "
                "temperature (energy cost) for each catalyst loading, at the sidebar pressure.")
    target = st.slider("Target expected yield (%)", 60.0, 99.0, 95.0, 0.5)
    Ts = np.linspace(*DOMAIN["temperature"], 251)
    Cs = np.linspace(*DOMAIN["catalyst_concentration"], 99)
    TT, CC = np.meshgrid(Ts, Cs)
    grid = np.c_[TT.ravel(), np.full(TT.size, P), CC.ravel()]
    Z = model.predict(grid).reshape(TT.shape)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    cf = ax.contourf(TT, CC, Z, levels=np.arange(20, 101, 5), cmap="viridis")
    ax.contour(TT, CC, Z, levels=[target], colors="white", linewidths=2)
    ax.plot(T, C, "r*", ms=14, label="current conditions")
    ax.set_xlabel(LABELS["temperature"])
    ax.set_ylabel(LABELS["catalyst_concentration"])
    ax.set_title(f"Predicted yield at P = {P:.1f} atm (white line = {target:.1f} %)")
    ax.legend(loc="lower right")
    fig.colorbar(cf, label="Yield (%)")
    fig.tight_layout()
    st.pyplot(fig)

    ok = Z >= target
    rows = []
    for i in np.linspace(0, len(Cs) - 1, 8).astype(int):
        j = np.argmax(ok[i]) if ok[i].any() else None
        rows.append({"Catalyst (mol/L)": f"{Cs[i]:.2f}",
                     "Min. temperature (°C)": f"{Ts[j]:.0f}" if j is not None else "not reachable"})
    st.table(rows)

# -------------------- Model comparison --------------------
with tab_models:
    st.markdown("5-fold cross-validation on 80 % of the data; final scores on an untouched "
                "20 % hold-out set. Run-to-run noise in the data is σ ≈ "
                f"{metrics['tobit_physical']['noise_sigma']:.1f} %, so no model can get "
                "much below that RMSE.")
    st.table([{"Model": n, "CV RMSE": f"{r['cv_rmse_mean']:.3f} ± {r['cv_rmse_std']:.3f}",
               "Test RMSE": f"{r['test_rmse']:.3f}", "Test MAE": f"{r['test_mae']:.3f}",
               "Test R²": f"{r['test_r2']:.4f}"} for n, r in metrics["models"].items()])
    p = metrics["tobit_physical"]
    st.markdown(f"**Tobit model in physical units:** yield* = {p['intercept']:.2f} "
                f"+ {p['slope_per_degC']:.4f}·T + {p['slope_per_atm']:.3f}·P "
                f"+ {p['slope_per_mol_L']:.2f}·C,  with observed yield = min(yield*, 100).")
