"""Reproducible input-quality summary for growth experiments."""
from dataclasses import dataclass
import pandas as pd
from .data_loader import GrowthData

@dataclass
class QualityReport:
    summary: pd.DataFrame
    by_time: pd.DataFrame
    warnings: list[str]


def quality_report(data: GrowthData) -> QualityReport:
    times = data.mean['time'].astype(float)
    counts = data.mean['n'].astype(int)
    values = data.mean['mean_weight'].astype(float)
    report = pd.DataFrame({
        'time': times.to_numpy(),
        'n': counts.to_numpy(),
        'mean_weight': values.to_numpy(),
    })
    report['missing_individuals'] = 0
    if data.format_name == 'A':
        possible = len(data.raw.columns) - 1
        report['missing_individuals'] = possible - counts.to_numpy()
    warnings = list(data.warnings)
    if times.diff().iloc[1:].nunique() > 1:
        warnings.append('Interval waktu tidak seragam; regresi tetap menggunakan waktu numerik aktual.')
    if data.format_name == 'B':
        warnings.append('Format B tidak memuat variasi individu; verifikasi konstanta C tidak tersedia.')
    summary = pd.DataFrame([{
        'format': data.format_name,
        'time_points': len(report),
        'observations': int(counts.sum()),
        'min_replicates': int(counts.min()),
        'max_replicates': int(counts.max()),
        'balanced': bool(counts.nunique() == 1),
        'missing_cells': int(report['missing_individuals'].sum()),
        'first_time': float(times.iloc[0]),
        'last_time': float(times.iloc[-1]),
    }])
    return QualityReport(summary, report, warnings)
