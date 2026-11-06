import numpy as np
import pandas as pd
import pytest

from core.data_loader import normalize
from core.models import logistic, gompertz, richards
from core.fitting import FitError, fit_model


def make_data(model="Logistic", unbalanced=False, noise=True):
    times = np.arange(0, 35, 2.0)
    x0 = 40.0
    curves = {
        "Logistic": logistic(times, 1400, 0.23, x0),
        "Gompertz": gompertz(times, 1400, 0.14, x0),
        "Richards": richards(times, 1400, 0.4, 0.24, x0),
    }
    y = curves[model]
    rng = np.random.default_rng(123)
    matrix = y[:, None] + (rng.normal(0, 3.0, (len(times), 6)) if noise else np.zeros((len(times), 6)))
    if unbalanced:
        matrix[4, 0] = np.nan
        matrix[10, 2:4] = np.nan
    frame = pd.DataFrame(matrix, columns=[f"Individu_{i}" for i in range(6)])
    frame.insert(0, "time", times)
    return normalize(frame, "A")


def test_three_models_initial_condition():
    for name, f in [("Logistic", lambda t: logistic(t, 1000, 0.2, 35)),
                    ("Gompertz", lambda t: gompertz(t, 1000, 0.2, 35)),
                    ("Richards", lambda t: richards(t, 1000, 0.5, 0.2, 35))]:
        assert f(np.array([0.]))[0] == pytest.approx(35.0), name


def test_richards_logistic_and_gompertz_limits():
    t = np.linspace(0, 30, 20)
    assert np.allclose(richards(t, 1000, 1, 0.2, 35), logistic(t, 1000, 0.2, 35))
    assert np.allclose(richards(t, 1000, 1e-5, 0.2 / 1e-5, 35), gompertz(t, 1000, 0.2, 35), rtol=1e-4)


@pytest.mark.parametrize("model", ["Logistic", "Gompertz", "Richards"])
def test_balanced_raw_and_mean_equivalent(model):
    d = make_data(model)
    full = fit_model(d, model, "raw")
    mean = fit_model(d, model, "mean")
    for p in full.parameters:
        assert full.parameters[p] == pytest.approx(mean.parameters[p], rel=2e-3, abs=1e-3)
    assert np.allclose(full.predicted, mean.predicted, rtol=1e-5, atol=0.01)


@pytest.mark.parametrize("model", ["Logistic", "Gompertz", "Richards"])
def test_unbalanced_raw_vs_weighted_equivalent(model):
    d = make_data(model, unbalanced=True)
    full = fit_model(d, model, "raw")
    weighted = fit_model(d, model, "weighted_mean")
    assert np.allclose(full.predicted, weighted.predicted, atol=0.02)
    assert np.allclose(list(full.parameters.values()), list(weighted.parameters.values()), rtol=3e-3, atol=1e-3)


def test_mean_only_rejects_raw():
    d = make_data()
    mean_only = normalize(d.mean.copy(), "B")
    with pytest.raises(FitError, match="Data individu"):
        fit_model(mean_only, "Logistic", "raw")


def test_exact_logistic_parameters():
    d = make_data("Logistic", noise=False)
    f = fit_model(d, "Logistic")
    assert f.parameters["K"] == pytest.approx(1400, rel=1e-4)
    assert f.parameters["r"] == pytest.approx(0.23, rel=1e-4)
