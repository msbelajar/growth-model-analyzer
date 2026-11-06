"""Comparable mean-based diagnostics and tidy exports for fitted growth models."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .fitting import FitResult


@dataclass
class DiagnosticResult:
    model: str
    table: pd.DataFrame
    residual_mean: float
    residual_sd: float
    mae: float
    max_abs_residual: float


def diagnostic_table(fit: FitResult) -> DiagnosticResult:
    """Diagnostics over time means excluding the fixed initial observation.

    This is the SAME assessment sample used for the Stage 3 RMSE/R2.
    Raw-individual residuals are intentionally not mixed into these values.
    """
    time = np.asarray(fit.time[1:], dtype=float)
    obs = np.asarray(fit.means[1:], dtype=float)
    pred = np.asarray(fit.predicted[1:], dtype=float)
    counts = np.asarray(fit.counts[1:], dtype=float)
    if not (len(time) == len(obs) == len(pred) == len(counts)) or len(time) == 0:
        raise ValueError("Data hasil fitting tidak memiliki dimensi diagnostik yang konsisten.")
    resid = obs - pred
    table = pd.DataFrame({
        "time": time, "n": counts.astype(int), "observed_mean": obs,
        "predicted_mean": pred, "residual": resid,
        "abs_residual": np.abs(resid),
    })
    return DiagnosticResult(
        model=fit.model, table=table,
        residual_mean=float(np.mean(resid)),
        residual_sd=float(np.std(resid, ddof=1)) if len(resid) > 1 else float("nan"),
        mae=float(np.mean(np.abs(resid))),
        max_abs_residual=float(np.max(np.abs(resid))),
    )


def comparison_table(results: list[FitResult]) -> pd.DataFrame:
    """One row per successful fit, with shared evaluation basis."""
    records = []
    for fit in results:
        diagnostic = diagnostic_table(fit)
        records.append({
            "Model": fit.model,
            "Basis fitting": fit.basis,
            "K": fit.parameters.get("K", np.nan),
            "a": fit.parameters.get("a", np.nan),
            "r": fit.parameters.get("r", np.nan),
            "RMSE mean": fit.rmse,
            "MAE mean": diagnostic.mae,
            "R² mean": fit.r2,
            "SSE mean": fit.sse_mean,
            "AIC mean": fit.aic_mean,
            "Residual mean": diagnostic.residual_mean,
        })
    return pd.DataFrame(records)


def prediction_table(results: list[FitResult]) -> pd.DataFrame:
    """One row per time with each model's predicted curve."""
    if not results:
        return pd.DataFrame(columns=["time", "mean_weight", "n"])
    first = results[0]
    frame = pd.DataFrame({"time": first.time, "mean_weight": first.means, "n": first.counts.astype(int)})
    for fit in results:
        if not np.array_equal(fit.time, first.time):
            raise ValueError("Waktu hasil fitting tidak konsisten antar-model.")
        frame[f"predicted_{fit.model.lower()}"] = fit.predicted
    return frame
