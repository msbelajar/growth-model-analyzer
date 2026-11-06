import numpy as np
import pandas as pd
import pytest
from core.data_loader import normalize
from core.fitting import fit_model
from core.models import logistic
from core.reporting import comparison_table
from core.export import report_dict


def make_data():
    t = np.arange(0, 35, dtype=float)
    y = logistic(t, 2400, 0.16, 38)
    # deterministic, non-zero residual means to avoid degenerate log(0)
    noise = 3 * np.sin(0.7 * t)
    return normalize(pd.DataFrame({"time":t, "id1":y+noise-2, "id2":y+noise+2}), "A")


@pytest.mark.parametrize("model,n_params", [("Logistic",2),("Gompertz",2),("Richards",3)])
def test_aic_formula(model,n_params):
    fit = fit_model(make_data(), model, "mean")
    m = len(fit.time)-1
    expected = m * np.log(fit.sse_mean / m) + 2*n_params
    assert fit.aic_mean == pytest.approx(expected, abs=1e-6)


def test_aic_on_same_evaluation_data_across_bases():
    data=make_data()
    f1=fit_model(data,"Logistic","mean")
    f2=fit_model(data,"Logistic","raw")
    assert f1.aic_mean == pytest.approx(f2.aic_mean, abs=1e-3)
    assert "AIC mean" in comparison_table([f1,f2]).columns
    assert report_dict(data,[f1])["fits"][0]["aic_mean"] == pytest.approx(f1.aic_mean)
