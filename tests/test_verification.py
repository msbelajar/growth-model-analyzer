import numpy as np
import pandas as pd
import pytest

from core.data_loader import normalize
from core.models import logistic, gompertz, richards
from core.verification import verify_proposition_1, VerificationError


@pytest.fixture
def balanced():
    t = np.arange(0., 40., 2.)
    curves = logistic(t, 1400., .22, 40.)
    rng = np.random.default_rng(17)
    samples = curves[:, None] + rng.normal(0, 2, (len(t), 5))
    f = pd.DataFrame(samples, columns=[f'Individual_{j}' for j in range(5)])
    f.insert(0, 'time', t)
    return normalize(f, 'A')


@pytest.mark.parametrize('model', ('Logistic', 'Gompertz', 'Richards'))
def test_balanced_identity_and_optimum(balanced, model):
    v = verify_proposition_1(balanced, model)
    assert v.is_balanced
    assert (v.identity_checks['Identitas berbobot'] == 'Sesuai').all()
    assert np.max(np.abs(v.identity_checks['d1 - N*d2 - C'])) < 1e-5
    assert np.allclose(v.raw_fit.predicted, v.mean_fit.predicted, atol=.1)
    assert np.allclose(v.raw_fit.predicted, v.weighted_fit.predicted, atol=.1)
    assert len(v.prediction_comparison) == 20


def test_unbalanced_weighting_identity(balanced):
    raw = balanced.raw.copy()
    raw.loc[4, 'Individual_0'] = np.nan
    raw.loc[6, 'Individual_1'] = np.nan
    d = normalize(raw, 'A')
    v = verify_proposition_1(d, 'Logistic')
    assert not v.is_balanced
    assert v.identity_checks['d1 - N*d2 - C'].isna().all()
    assert (v.identity_checks['Identitas berbobot'] == 'Sesuai').all()
    assert np.allclose(v.raw_fit.predicted, v.weighted_fit.predicted, atol=.1)


def test_mean_input_cannot_verify(balanced):
    b = normalize(balanced.mean.copy(), 'B')
    with pytest.raises(VerificationError, match='Format A'):
        verify_proposition_1(b, 'Logistic')


def test_constant_is_not_raw_sse(balanced):
    v = verify_proposition_1(balanced, 'Logistic')
    row = v.summary.iloc[0]
    assert row['C'] > 0
    assert row['SSE data lengkap'] == pytest.approx(row['SSE berbobot'] + row['C'], abs=1e-5)
