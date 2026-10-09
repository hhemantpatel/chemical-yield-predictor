"""Train and evaluate every model, then save them and write reports/.

    python train.py              # CSV data (default)
    YIELD_DATA_SOURCE=mysql python train.py

Evaluation: 5-fold cross-validation on 80 % of the data for model selection,
then a single untouched 20 % hold-out set for the final numbers.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

from yieldpred.data import FEATURES, ROOT, TARGET, YIELD_CAP, load
from yieldpred.models import build_models, physical_coefficients

SEED = 42
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def slug(name):
    return name.split(" (")[0].lower().replace(" ", "_")


def main():
    df = load()
    X, y = df[FEATURES].to_numpy(), df[TARGET].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"{len(df)} rows; {np.mean(y >= YIELD_CAP):.1%} at the {YIELD_CAP:.0f} % cap")

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"rmse": "neg_root_mean_squared_error",
               "mae": "neg_mean_absolute_error", "r2": "r2"}
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    results = {}
    for name, model in build_models(SEED).items():
        cvr = cross_validate(model, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        results[name] = {
            "cv_rmse_mean": float(-cvr["test_rmse"].mean()),
            "cv_rmse_std": float(cvr["test_rmse"].std()),
            "cv_mae_mean": float(-cvr["test_mae"].mean()),
            "cv_r2_mean": float(cvr["test_r2"].mean()),
            "test_rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
            "test_mae": float(mean_absolute_error(y_te, pred)),
            "test_r2": float(r2_score(y_te, pred)),
            "file": f"{slug(name)}.joblib",
        }
        joblib.dump(model, MODELS_DIR / results[name]["file"])
        r = results[name]
        print(f"{name:26s} CV RMSE {r['cv_rmse_mean']:.3f} ± {r['cv_rmse_std']:.3f}"
              f" | test RMSE {r['test_rmse']:.3f}  MAE {r['test_mae']:.3f}  R² {r['test_r2']:.4f}")

    tobit = joblib.load(MODELS_DIR / results["Tobit (censored linear)"]["file"])
    intercept, slopes, sigma = physical_coefficients(tobit)
    physics = {
        "intercept": float(intercept),
        "slope_per_degC": float(slopes[0]),
        "slope_per_atm": float(slopes[1]),
        "slope_per_mol_L": float(slopes[2]),
        "noise_sigma": float(sigma),
    }
    print("Tobit latent model: yield* = {intercept:.2f} + {slope_per_degC:.4f}·T"
          " + {slope_per_atm:.3f}·P + {slope_per_mol_L:.2f}·C,  σ = {noise_sigma:.2f}"
          .format(**physics))

    best = min(results, key=lambda n: results[n]["cv_rmse_mean"])
    summary = {"n_rows": len(df), "frac_at_cap": float(np.mean(y >= YIELD_CAP)),
               "best_model": best, "models": results, "tobit_physical": physics}
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    write_markdown(summary)
    print(f"best by CV RMSE: {best}; reports written to {REPORTS_DIR}/")


def write_markdown(summary):
    lines = ["| Model | CV RMSE (5-fold) | Test RMSE | Test MAE | Test R² |",
             "| --- | --- | --- | --- | --- |"]
    for name, r in summary["models"].items():
        mark = " **(best)**" if name == summary["best_model"] else ""
        lines.append(f"| {name}{mark} | {r['cv_rmse_mean']:.3f} ± {r['cv_rmse_std']:.3f} "
                     f"| {r['test_rmse']:.3f} | {r['test_mae']:.3f} | {r['test_r2']:.4f} |")
    Path(REPORTS_DIR / "metrics.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
