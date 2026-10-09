import numpy as np
import pytest

from yieldpred.data import DOMAIN, FEATURES, TARGET, load_csv, out_of_domain
from yieldpred.models import build_models
from yieldpred.tobit import TobitRegressor


def test_dataset_loads_clean():
    df = load_csv()
    assert len(df) == 10_000
    assert list(df.columns) == FEATURES + [TARGET]
    assert df[TARGET].between(0, 100).all()
    for feat, (lo, hi) in DOMAIN.items():
        assert df[feat].min() >= lo - 1e-6 and df[feat].max() <= hi + 1e-6


def test_tobit_recovers_known_censored_line():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(4000, 2))
    y_star = 78 + 25 * X[:, 0] + 10 * X[:, 1] + rng.normal(0, 3, len(X))
    y = np.minimum(y_star, 100)
    assert (y == 100).mean() > 0.1  # the test is only meaningful with censoring

    m = TobitRegressor(cap=100).fit(X, y)
    assert m.intercept_ == pytest.approx(78, abs=0.5)
    assert m.coef_ == pytest.approx([25, 10], abs=0.7)
    assert m.sigma_ == pytest.approx(3, abs=0.2)

    pred = m.predict(X)
    assert pred.max() <= 100
    p = m.prob_full_conversion(X)
    assert ((0 <= p) & (p <= 1)).all()


def test_all_models_predict_within_physical_range():
    df = load_csv().sample(1500, random_state=1)
    X, y = df[FEATURES].to_numpy(), df[TARGET].to_numpy()
    extreme = np.array([[300, 10, 0.5], [50, 1, 0.01]])
    for name, model in build_models().items():
        model.fit(X, y)
        pred = model.predict(extreme)
        assert ((0 <= pred) & (pred <= 100)).all(), name


def test_out_of_domain_flags():
    assert out_of_domain(150, 5, 0.25) == []
    assert out_of_domain(20, 5, 0.9) == ["temperature", "catalyst_concentration"]
