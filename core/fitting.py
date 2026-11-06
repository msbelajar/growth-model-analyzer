"""Bounded nonlinear least-squares fitting and model comparison.

For datasets with equal counts at every time, ordinary least squares (OLS)
on individual observations has exactly the same minimizers as unweighted
OLS on time-point means. For unequal counts, a weighted fit on the means
with weights n_i preserves the raw OLS minimizers.
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import least_squares

from .data_loader import GrowthData
from .models import MODEL_NAMES, PARAMETER_NAMES, evaluate

FitBasis = Literal["mean", "weighted_mean", "raw"]


class FitError(ValueError):
    pass


@dataclass
class FitResult:
    model: str
    basis: str
    parameters: dict[str, float]
    x0: float
    t0: float
    time: np.ndarray
    means: np.ndarray
    counts: np.ndarray
    predicted: np.ndarray
    rmse: float
    r2: float
    sse_mean: float
    sse_objective: float
    aic_mean: float
    success: bool
    message: str


def _bounds_and_starts(model: str, time: np.ndarray, means: np.ndarray, x0: float):
    span = float(time[-1] - time[0])
    if span <= 0:
        raise FitError("Rentang waktu pengamatan harus positif.")
    ymax = float(np.max(means))
    # Constrain K to above initial weight; do NOT force K above all observations:
    # noisy measurements can exceed the asymptote.
    k_lower = x0 * (1.0 + 1e-7)
    k_upper = max(ymax * 100.0, x0 * 100.0, k_lower * 2)
    r_lower = 1e-8 / span
    r_upper = 100.0 / span
    base_k = max(ymax * 1.15, x0 * 1.5)
    k_candidates = [base_k, ymax * 1.8, ymax * 3.0]
    rate_candidates = [1.0 / span, 3.0 / span, 7.0 / span]
    if model == "Richards":
        lower = [k_lower, 0.005, r_lower]
        upper = [k_upper, 10.0, r_upper]
        starts = [[k, a, r] for k in k_candidates for a in (0.1, 1.0, 2.0) for r in rate_candidates]
    else:
        lower = [k_lower, r_lower]
        upper = [k_upper, r_upper]
        starts = [[k, r] for k in k_candidates for r in rate_candidates]
    bounds = (np.array(lower, float), np.array(upper, float))
    starts = [np.clip(np.array(start, float), bounds[0] + 1e-10, bounds[1] - 1e-10) for start in starts]
    return bounds, starts


def fit_model(data: GrowthData, model: str, basis: FitBasis = "mean") -> FitResult:
    """Estimate the model using the selected data source.

    We fix x0 at the average at the first time and omit that time from the
    residual objective (consistent with the paper). Reported RMSE/R2
    are consistently computed from the *remaining time means* for model
    comparison, regardless of fitting basis. They are in-sample diagnostics.

    """
    if model not in MODEL_NAMES:
        raise FitError(f"Model tidak dikenal: {model}")
    if basis not in ("mean", "weighted_mean", "raw"):
        raise FitError(f"Basis fitting tidak dikenal: {basis}")
    if basis == "raw" and data.long is None:
        raise FitError("Data individu diperlukan untuk fitting langsung pada data lengkap.")

    mean_table = data.mean
    time = mean_table["time"].to_numpy(dtype=float)
    means = mean_table["mean_weight"].to_numpy(dtype=float)
    counts = mean_table["n"].to_numpy(dtype=float)
    x0, t0 = float(means[0]), float(time[0])
    shifted = time - t0
    if len(time) < len(PARAMETER_NAMES[model]) + 2:
        raise FitError(f"Model {model} memerlukan minimal {len(PARAMETER_NAMES[model]) + 2} titik waktu (termasuk waktu awal).")
    if np.any(means <= 0) or x0 <= 0:
        raise FitError("Bobot data harus positif.")

    # First time is a fixed initial condition, not a fitting residual.
    fit_times = shifted[1:]
    fit_means = means[1:]
    fit_counts = counts[1:]
    if basis == "raw":
        raw = data.long
        mask = raw["time"].to_numpy(dtype=float) != t0
        target = raw.loc[mask, "weight"].to_numpy(dtype=float)
        input_t = raw.loc[mask, "time"].to_numpy(dtype=float) - t0
        weights = np.ones_like(target)
    else:
        target = fit_means
        input_t = fit_times
        weights = np.sqrt(fit_counts) if basis == "weighted_mean" else np.ones_like(target)
    bounds, starts = _bounds_and_starts(model, time, means, x0)

    def residuals(p):
        try:
            pred = evaluate(model, input_t, p, x0)
            if not np.isfinite(pred).all():
                return np.full(len(target), 1e15)
            return (pred - target) * weights
        except (FloatingPointError, OverflowError, ValueError):
            return np.full(len(target), 1e15)

    best = None
    for start in starts:
        try:
            opt = least_squares(residuals, start, bounds=bounds,
                                max_nfev=3000, xtol=1e-11, ftol=1e-11, gtol=1e-11,
                                x_scale="jac")
        except (ValueError, FloatingPointError):
            continue
        if opt.success and np.isfinite(opt.cost) and (best is None or opt.cost < best.cost):
            best = opt
    if best is None:
        raise FitError(f"Optimasi {model} tidak konvergen. Periksa data atau rentang waktu.")

    predicted = np.asarray(evaluate(model, shifted, best.x, x0), dtype=float)
    if not np.isfinite(predicted).all():
        raise FitError("Prediksi model mengandung angka tidak valid.")
    # Uniform time-mean metrics are comparable across all 3 models.
    mean_resid = means[1:] - predicted[1:]
    sse = float(np.sum(mean_resid**2))
    n_time = len(mean_resid)
    rmse = float(np.sqrt(sse / n_time))
    # Gaussian errors with common unknown variance on time-point means.
    # Additive constants independent of the model (including the variance
    # parameter penalty) are omitted, so model rankings are unchanged.
    # For raw/weighted fits, this is a mean-data evaluation score, NOT
    # the maximized-likelihood AIC of the raw/weighted fitting objective.
    parameter_count = len(PARAMETER_NAMES[model])
    aic_mean = float(n_time * np.log(sse / n_time) + 2 * parameter_count) if sse > 0 else float("nan")
    tss = float(np.sum((means[1:] - np.mean(means[1:]))**2))
    r2 = float(1 - sse / tss) if tss > 0 else float("nan")
    return FitResult(model, basis, dict(zip(PARAMETER_NAMES[model], map(float, best.x))),
                     x0, t0, time, means, counts, predicted, rmse, r2, sse,
                     float(2 * best.cost), aic_mean, bool(best.success), str(best.message))


def fit_all(data: GrowthData, basis: FitBasis = "mean") -> tuple[list[FitResult], dict[str, str]]:
    results = []
    errors = {}
    for name in MODEL_NAMES:
        try:
            results.append(fit_model(data, name, basis))
        except FitError as exc:
            errors[name] = str(exc)
    return results, errors
