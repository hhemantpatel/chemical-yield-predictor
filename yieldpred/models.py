"""Model zoo. Every model is a Pipeline that owns its scaler, so a saved
model can never be separated from the scaling it was trained with."""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .data import YIELD_CAP
from .tobit import TobitRegressor


def _clip(y):
    return np.clip(y, 0.0, YIELD_CAP)


def _identity(y):
    return y


def _capped(model):
    """Clip predictions to the physical 0-100 % range."""
    return TransformedTargetRegressor(regressor=model, inverse_func=_clip,
                                      func=_identity, check_inverse=False)


def build_models(seed=42):
    return {
        "Linear (OLS)": _capped(make_pipeline(StandardScaler(), LinearRegression())),
        "Tobit (censored linear)": make_pipeline(StandardScaler(),
                                                 TobitRegressor(cap=YIELD_CAP)),
        # The three original models, with the original hyperparameters.
        "KNN (k=20)": _capped(make_pipeline(StandardScaler(),
                                            KNeighborsRegressor(n_neighbors=20))),
        "SVR (RBF)": _capped(make_pipeline(StandardScaler(),
                                           SVR(kernel="rbf", C=10, gamma="scale", epsilon=1))),
        "Neural net (MLP)": _capped(make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 64), activation="tanh",
                         alpha=1e-3, learning_rate_init=1e-3, max_iter=2000,
                         early_stopping=True, random_state=seed))),
        "Gradient boosting": _capped(HistGradientBoostingRegressor(random_state=seed)),
    }


def tobit_of(pipeline):
    """The fitted TobitRegressor inside a pipeline, plus its scaler."""
    return pipeline[-1], pipeline[0]


def physical_coefficients(pipeline):
    """Tobit slopes in physical units (% yield per unit of each input)."""
    tobit, scaler = tobit_of(pipeline)
    slopes = tobit.coef_ / scaler.scale_
    intercept = tobit.intercept_ - (tobit.coef_ * scaler.mean_ / scaler.scale_).sum()
    return intercept, slopes, tobit.sigma_
