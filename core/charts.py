"""Interactive Plotly graphics and independent high-resolution PNG via Matplotlib."""
from io import BytesIO

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data_loader import GrowthData
from .fitting import FitResult
from .models import evaluate, PARAMETER_NAMES
from .reporting import diagnostic_table

COLORS = {"Logistic": "#2563eb", "Gompertz": "#0f766e", "Richards": "#a855f7"}


def _dense_curve(fit: FitResult, n: int = 350) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(float(fit.time[0]), float(fit.time[-1]), n)
    params = [fit.parameters[name] for name in PARAMETER_NAMES[fit.model]]
    y = evaluate(fit.model, t - fit.t0, params, fit.x0)
    return t, np.asarray(y, dtype=float)


def growth_figure(data: GrowthData, results: list[FitResult], show_individuals: bool = True) -> go.Figure:
    fig = go.Figure()
    if show_individuals and data.long is not None:
        fig.add_trace(go.Scatter(
            x=data.long["time"], y=data.long["weight"], mode="markers",
            name="Data individu", marker={"color": "rgba(100,116,139,0.27)", "size": 5},
            hovertemplate="Waktu=%{x}<br>Bobot=%{y:.3f}<extra>Individu</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=data.mean["time"], y=data.mean["mean_weight"], mode="markers",
        name="Rata-rata pengamatan", marker={"color": "#e11d48", "size": 8, "symbol": "circle"},
        hovertemplate="Waktu=%{x}<br>Rata-rata=%{y:.3f}<extra></extra>",
    ))
    for fit in results:
        t, y = _dense_curve(fit)
        fig.add_trace(go.Scatter(
            x=t, y=y, name=fit.model, mode="lines",
            line={"width": 3, "color": COLORS[fit.model]},
            hovertemplate="Waktu=%{x:.3f}<br>Prediksi=%{y:.3f}<extra>" + fit.model + "</extra>",
        ))
    fig.update_layout(title="Kurva pertumbuhan dan data pengamatan", xaxis_title="Waktu",
                      yaxis_title="Bobot", template="plotly_white", hovermode="x unified",
                      margin={"l": 30, "r": 20, "t": 70, "b": 40}, legend={"orientation": "h", "y": -0.23})
    return fig


def residual_figure(results: list[FitResult]) -> go.Figure:
    """Residual by time; zero line makes bias and trends easy to detect."""
    fig = go.Figure()
    for fit in results:
        d = diagnostic_table(fit).table
        fig.add_trace(go.Scatter(
            x=d["time"], y=d["residual"], mode="lines+markers", name=fit.model,
            line={"color": COLORS[fit.model], "width": 2},
            hovertemplate="Waktu=%{x}<br>Residual=%{y:.3f}<extra>" + fit.model + "</extra>",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="#64748b")
    fig.update_layout(title="Residual rata-rata terhadap waktu", xaxis_title="Waktu",
                      yaxis_title="Residual (observasi − prediksi)", template="plotly_white",
                      margin={"l": 30, "r": 20, "t": 60, "b": 40})
    return fig


def residual_scatter_figure(results: list[FitResult]) -> go.Figure:
    fig = go.Figure()
    for fit in results:
        d = diagnostic_table(fit).table
        fig.add_trace(go.Scatter(
            x=d["predicted_mean"], y=d["residual"], mode="markers", name=fit.model,
            marker={"color": COLORS[fit.model], "size": 8, "opacity": 0.75},
            customdata=np.asarray(d["time"]),
            hovertemplate="Prediksi=%{x:.3f}<br>Residual=%{y:.3f}<br>Waktu=%{customdata}<extra>" + fit.model + "</extra>",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="#64748b")
    fig.update_layout(title="Residual terhadap nilai prediksi", xaxis_title="Bobot prediksi",
                      yaxis_title="Residual (observasi − prediksi)", template="plotly_white",
                      margin={"l": 30, "r": 20, "t": 60, "b": 40})
    return fig


def criteria_figure(results: list[FitResult], metric: str = "RMSE") -> go.Figure:
    if metric not in ("RMSE", "MAE", "AIC"):
        raise ValueError("Metrik harus RMSE, MAE atau AIC.")
    labels = [f.model for f in results]
    vals = [f.rmse if metric == "RMSE" else diagnostic_table(f).mae if metric == "MAE" else f.aic_mean for f in results]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=[COLORS[x] for x in labels],
                           text=[f"{v:,.3f}" for v in vals], textposition="auto"))
    fig.update_layout(title=f"Perbandingan {metric} pada rata-rata pengamatan",
                      xaxis_title="Model", yaxis_title=metric,
                      template="plotly_white", margin={"l": 30, "r": 20, "t": 60, "b": 40})
    return fig


def publication_png(data: GrowthData, results: list[FitResult], dpi: int = 300,
                    show_individuals: bool = True) -> bytes:
    """Static publication image without Kaleido/Chrome dependencies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    if show_individuals and data.long is not None:
        ax.scatter(data.long["time"], data.long["weight"], c="#94a3b8", alpha=0.22,
                   s=8, label="Individu", rasterized=True)
    ax.scatter(data.mean["time"], data.mean["mean_weight"], c="#e11d48", s=25,
               label="Rata-rata", zorder=4)
    for fit in results:
        tt, yy = _dense_curve(fit)
        ax.plot(tt, yy, label=fit.model, color=COLORS[fit.model], linewidth=2.4)
    ax.set(xlabel="Waktu", ylabel="Bobot", title="Kurva Pertumbuhan Nonlinear")
    ax.grid(True, color="#e2e8f0", linewidth=0.7)
    ax.legend(loc="best", frameon=False)
    output = BytesIO()
    fig.savefig(output, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    return output.getvalue()


def residual_png(results: list[FitResult], dpi: int = 300) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    for fit in results:
        d = diagnostic_table(fit).table
        ax.plot(d["time"], d["residual"], marker="o", markersize=3,
                linewidth=1.5, color=COLORS[fit.model], label=fit.model)
    ax.axhline(0, linestyle="--", linewidth=1.0, color="#64748b")
    ax.set(xlabel="Waktu", ylabel="Residual (observasi − prediksi)",
           title="Residual Rata-rata terhadap Waktu")
    ax.grid(True, color="#e2e8f0", linewidth=0.7)
    ax.legend(frameon=False)
    output = BytesIO()
    fig.savefig(output, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    return output.getvalue()
