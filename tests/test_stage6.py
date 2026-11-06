import json
import importlib.util
import numpy as np
import pandas as pd
import pytest
from core.data_loader import normalize, read_excel, read_pasted
from core.fitting import fit_model
from core.models import logistic
from core.quality import quality_report
from core.export import report_json


def data_a():
    t = np.arange(0, 30, 2, dtype=float)
    theoretical = logistic(t, K=850., r=.23, x0=35.)
    df = pd.DataFrame({
        'time': t,
        'animal_1': theoretical + 1.,
        'animal_2': theoretical - 1.,
        'animal_3': theoretical,
    })
    return df


def test_quality_balanced():
    q = quality_report(normalize(data_a(), 'A'))
    assert q.summary.iloc[0]['balanced']
    assert q.summary.iloc[0]['missing_cells'] == 0


def test_quality_missing_and_unbalanced():
    df = data_a()
    df.loc[2, 'animal_3'] = np.nan
    q = quality_report(normalize(df, 'A'))
    assert not q.summary.iloc[0]['balanced']
    assert q.summary.iloc[0]['missing_cells'] == 1


def test_report_json_with_aic():
    d = normalize(data_a(), 'A')
    fit = fit_model(d, 'Logistic', 'mean')
    payload = report_json(d, [fit])
    report = json.loads(payload)
    assert report['fits'][0]['parameters']['K'] == pytest.approx(850, abs=2)
    assert report['input']['time_points'] == 15
    assert report['fits'][0]['aic_mean'] == pytest.approx(fit.aic_mean)


def test_excel_example_end_to_end():
    if importlib.util.find_spec("python_calamine") is None:
        pytest.skip("python-calamine belum terpasang pada lingkungan pengujian ini")
    d = normalize(read_excel('examples/data_10_individu_50_hari.xlsx', sheet_name='Data_Individu'), 'A')
    assert d.format_name == 'A'
    assert len(d.mean) == 50
    assert (d.mean['n'] == 10).all()
    fit = fit_model(d, 'Logistic', 'mean')
    assert fit.success and np.isfinite(fit.rmse)


def test_text_and_excel_same_normalization():
    source = 'time\tA\tB\n0\t35\t37\n1\t40\t42\n2\t45\t47'
    d = normalize(read_pasted(source), 'A')
    assert list(d.mean['mean_weight']) == [36., 41., 46.]
