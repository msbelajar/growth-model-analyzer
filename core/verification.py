"""Numerical verification of mean-substituted least-squares identities.

The fitted first time point is fixed to its observed mean x0, so all
objectives and their within-time constant exclude that first time point.
No AIC or model-selection likelihood calculations are used here.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .data_loader import GrowthData
from .fitting import FitResult, fit_model, FitError
from .models import evaluate


class VerificationError(ValueError):
    pass


@dataclass
class VerificationResult:
    model: str
    is_balanced: bool
    n_by_time: np.ndarray
    raw_fit: FitResult
    mean_fit: FitResult
    weighted_fit: FitResult
    summary: pd.DataFrame
    parameter_comparison: pd.DataFrame
    identity_checks: pd.DataFrame
    prediction_comparison: pd.DataFrame
    numerical_tolerance: float


def _objective_terms(data: GrowthData, fit: FitResult) -> dict[str, float]:
    """Evaluate SSE for all three objectives at exactly one parameter vector."""
    times = data.mean['time'].to_numpy(float)
    means = data.mean['mean_weight'].to_numpy(float)
    counts = data.mean['n'].to_numpy(float)
    p = [fit.parameters[k] for k in (('K', 'a', 'r') if fit.model == 'Richards' else ('K', 'r'))]
    pred = np.asarray(evaluate(fit.model, times - fit.t0, p, fit.x0), float)
    errors = pred[1:] - means[1:]
    sse_mean = float(np.sum(errors ** 2))
    sse_weighted = float(np.dot(counts[1:], errors ** 2))

    # Explicitly compute raw-data SSE as an independent check.
    raw = data.long
    if raw is None:
        raise VerificationError('Data individu Format A diperlukan untuk menghitung SSE data lengkap.')
    raw_sub = raw.loc[raw['time'].to_numpy(float) != fit.t0]
    rp = np.asarray(evaluate(fit.model, raw_sub['time'].to_numpy(float) - fit.t0,
                             p, fit.x0), float)
    sse_raw = float(np.sum((rp - raw_sub['weight'].to_numpy(float)) ** 2))

    # Within-time sum of squares, using only fitted time points.
    joined = raw_sub.merge(data.mean[['time', 'mean_weight']], on='time', how='left', validate='many_to_one')
    within = float(np.sum((joined['weight'].to_numpy(float) - joined['mean_weight'].to_numpy(float)) ** 2))
    return dict(sse_raw=sse_raw, sse_mean=sse_mean,
                sse_weighted=sse_weighted, C=within,
                diff_weighted=sse_raw - sse_weighted - within,
                diff_balanced=(sse_raw - counts[1] * sse_mean - within)
                    if bool(np.all(counts[1:] == counts[1])) else float('nan'))


def verify_proposition_1(data: GrowthData, model: str, *, rtol: float = 1e-8) -> VerificationResult:
    if data.long is None:
        raise VerificationError('Verifikasi dengan data lengkap hanya tersedia untuk Format A (data individu).')
    counts = data.mean['n'].to_numpy(int)
    balanced = bool(np.all(counts[1:] == counts[1]))
    raw_fit = fit_model(data, model, 'raw')
    mean_fit = fit_model(data, model, 'mean')
    weighted_fit = fit_model(data, model, 'weighted_mean')
    fits = [('Data individu', raw_fit), ('Rata-rata', mean_fit), ('Rata-rata berbobot', weighted_fit)]

    summary_rows = []
    identity_rows = []
    params = []
    for name, fit in fits:
        terms = _objective_terms(data, fit)
        scale = max(1.0, abs(terms['sse_raw']), abs(terms['sse_weighted']) + abs(terms['C']))
        valid = abs(terms['diff_weighted']) <= rtol * scale
        summary_rows.append(dict(Basis=name, K=fit.parameters['K'],
                                 a=fit.parameters.get('a', float('nan')),
                                 r=fit.parameters['r'],
                                 **{'SSE data lengkap': terms['sse_raw'],
                                    'SSE rata-rata': terms['sse_mean'],
                                    'SSE berbobot': terms['sse_weighted'],
                                    'C': terms['C']}))
        identity_rows.append(dict(Basis=name,
                                  **{'d1 - dw - C': terms['diff_weighted'],
                                     'd1 - N*d2 - C': terms['diff_balanced'],
                                     'Identitas berbobot': 'Sesuai' if valid else 'Perlu diperiksa'}))
    for name in raw_fit.parameters:
        params.append({'Parameter': name,
                       'Data individu': raw_fit.parameters[name],
                       'Rata-rata': mean_fit.parameters[name],
                       'Rata-rata berbobot': weighted_fit.parameters[name],
                       'Selisih |raw-mean|': abs(raw_fit.parameters[name] - mean_fit.parameters[name]),
                       'Selisih |raw-weighted|': abs(raw_fit.parameters[name] - weighted_fit.parameters[name])})
    predictions = pd.DataFrame({
        'time': data.mean['time'].to_numpy(float),
        'mean_weight': data.mean['mean_weight'].to_numpy(float),
        'n': counts,
        'pred_raw': raw_fit.predicted,
        'pred_mean': mean_fit.predicted,
        'pred_weighted': weighted_fit.predicted,
    })
    predictions['abs_raw_minus_mean'] = (predictions.pred_raw - predictions.pred_mean).abs()
    predictions['abs_raw_minus_weighted'] = (predictions.pred_raw - predictions.pred_weighted).abs()
    return VerificationResult(model, balanced, counts, raw_fit, mean_fit, weighted_fit,
                              pd.DataFrame(summary_rows), pd.DataFrame(params),
                              pd.DataFrame(identity_rows), predictions, rtol)
