from io import BytesIO
import numpy as np
import pandas as pd
import pytest

from core.data_loader import normalize
from core.fitting import fit_model
from core.models import logistic
from core.reporting import comparison_table, diagnostic_table, prediction_table
from core.charts import growth_figure, residual_figure, residual_scatter_figure, criteria_figure, publication_png, residual_png

@pytest.fixture(scope="module")
def setup_results():
    t = np.arange(0, 40, 2)
    y = logistic(t, K=4000, r=0.18, x0=40)
    df = pd.DataFrame({"time": t, "Individu_01": y + 1,
                       "Individu_02": y - 1, "Individu_03": y})
    data = normalize(df, "A")
    results = [fit_model(data, "Logistic", "mean"), fit_model(data, "Gompertz", "mean"), fit_model(data, "Richards", "mean")]
    return data, results


def test_reporting_uses_same_time_points_as_fit(setup_results):
    data, results = setup_results
    for fit in results:
        d = diagnostic_table(fit)
        assert len(d.table) == len(data.mean) - 1
        assert d.table["time"].iloc[0] == data.mean["time"].iloc[1]
        assert np.isclose(np.sqrt(np.mean(d.table["residual"]**2)), fit.rmse)
        assert np.isclose(np.sum(d.table["residual"]**2), fit.sse_mean)


def test_comparison_and_predictions(setup_results):
    data, results = setup_results
    summary = comparison_table(results)
    pred = prediction_table(results)
    assert len(summary) == 3
    assert len(pred) == len(data.mean)
    assert {"predicted_logistic", "predicted_gompertz", "predicted_richards"}.issubset(pred.columns)
    assert summary["RMSE mean"].notna().all()
    assert "AIC mean" in summary.columns
    assert np.isfinite(summary["AIC mean"]).all()


def test_interactive_figures(setup_results):
    data, results = setup_results
    assert len(growth_figure(data, results).data) == 5
    assert len(growth_figure(data, results, show_individuals=False).data) == 4
    assert len(residual_figure(results).data) == 3
    assert len(residual_scatter_figure(results).data) == 3
    for metric in ("RMSE", "MAE", "AIC"):
        assert len(criteria_figure(results, metric).data) == 1
    with pytest.raises(ValueError):
        criteria_figure(results, "P")


def test_png_outputs_are_valid(setup_results):
    data, results = setup_results
    for blob in (publication_png(data, results, dpi=110), residual_png(results, dpi=110)):
        assert blob.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(blob) > 5000


def test_export_html_is_self_contained(setup_results):
    data, results = setup_results
    html = growth_figure(data, results).to_html(full_html=True, include_plotlyjs=True)
    assert "plotly" in html.lower()
    assert "Logistic" in html
