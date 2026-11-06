"""Machine-readable reproducibility reports with mean-data AIC."""
from datetime import datetime, timezone
import json
import math
from .data_loader import GrowthData
from .fitting import FitResult


def _safe_number(x):
    v = float(x)
    return v if math.isfinite(v) else None


def report_dict(data: GrowthData, fits: list[FitResult]) -> dict:
    return {
        'schema': 'growth-model-analyzer/v1',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'method': {
            'objective': 'nonlinear least squares',
            'initial_condition': 'x0 = mean at first time; first time excluded from fitting',
            'model_comparison': 'metrics on time-point means excluding the first time',
            'model_selection_rule': 'show RMSE, MAE, R2 and mean-data AIC; lower AIC among comparable mean-fits preferred',
            'aic_mean_definition': 'm*ln(SSE_mean/m) + 2*p, m excludes fixed initial time, p counts fitted growth parameters',
            'aic_mean_scope': 'Gaussian equal-variance time-mean evaluation; a diagnostic score, not raw-data likelihood AIC when basis is raw or weighted_mean',
            'note': 'These diagnostics do not measure predictive performance on new individuals.',
        },
        'input': {
            'format': data.format_name,
            'time_points': len(data.mean),
            'total_observations': int(data.mean['n'].sum()),
            'balanced': bool(data.is_balanced),
        },
        'fits': [{
            'model': fit.model,
            'basis': fit.basis,
            'parameters': {name: _safe_number(val) for name, val in fit.parameters.items()},
            'fixed_x0': _safe_number(fit.x0),
            'fixed_t0': _safe_number(fit.t0),
            'rmse_mean': _safe_number(fit.rmse),
            'r2_mean': _safe_number(fit.r2),
            'sse_mean': _safe_number(fit.sse_mean),
            'aic_mean': _safe_number(fit.aic_mean),
            'sse_objective': _safe_number(fit.sse_objective),
            'success': bool(fit.success),
            'message': fit.message,
        } for fit in fits],
    }


def report_json(data: GrowthData, fits: list[FitResult]) -> bytes:
    return json.dumps(report_dict(data, fits), ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
