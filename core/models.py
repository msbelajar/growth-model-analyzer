"""Growth models in mean_sub.pdf. Time is shifted so t=0 is first sample.

All parameters follow the paper's notation. x0 is fixed at the first
observed time-point mean (it is NOT fitted).
"""
import numpy as np

MODEL_NAMES = ("Logistic", "Gompertz", "Richards")
PARAMETER_NAMES = {"Logistic": ("K", "r"), "Gompertz": ("K", "r"), "Richards": ("K", "a", "r")}


def logistic(t, K, r, x0):
    t = np.asarray(t, dtype=float)
    return K / (1.0 + (K / x0 - 1.0) * np.exp(-r * t))


def gompertz(t, K, r, x0):
    t = np.asarray(t, dtype=float)
    return K * np.exp(np.log(x0 / K) * np.exp(-r * t))


def richards(t, K, a, r, x0):
    """Richards: K/[1+((K/x0)^a -1) exp(-a r t)]^(1/a).

    Uses expm1/log1p for stability near a=0. In the PDF parametrization
    exp(-a*r*t), the limit a -> 0 with fixed r is constant x0;
    Gompertz is recovered if a*r is held fixed as a -> 0.
    """
    t = np.asarray(t, dtype=float)
    log_ratio = np.log(K / x0)
    z = np.expm1(a * log_ratio) * np.exp(-a * r * t)
    return K * np.exp(-np.log1p(z) / a)


def evaluate(model, t, params, x0):
    if model == "Logistic":
        return logistic(t, params[0], params[1], x0)
    if model == "Gompertz":
        return gompertz(t, params[0], params[1], x0)
    if model == "Richards":
        return richards(t, params[0], params[1], params[2], x0)
    raise ValueError(f"Model tidak dikenal: {model}")
