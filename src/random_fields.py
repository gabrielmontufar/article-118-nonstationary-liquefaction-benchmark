"""Minimal vertical random-field diagnostics for CPT-derived variables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def empirical_variogram(z: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    rows = []
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            rows.append((abs(z[i] - z[j]), 0.5 * (y[i] - y[j]) ** 2))
    pair = pd.DataFrame(rows, columns=["h_m", "gamma"])
    if pair.empty:
        return pair
    pair["bin"] = pd.qcut(pair["h_m"], q=min(n_bins, len(pair)), duplicates="drop")
    return pair.groupby("bin", observed=False).agg(h_m=("h_m", "mean"), gamma=("gamma", "mean"), n_pairs=("gamma", "size")).reset_index(drop=True)


def fit_exponential_variogram(z: np.ndarray, y: np.ndarray) -> dict:
    emp = empirical_variogram(z, y)
    if emp.empty:
        return {"model_type": "exponential", "nugget": 0.0, "sill": float(np.var(y)), "theta_z_m": 1.0, "n_pairs": 0, "rmse": float("nan")}
    sill = max(float(np.nanvar(y)), 1e-6)
    theta_grid = np.linspace(0.25, 20.0, 120)
    best = None
    for theta in theta_grid:
        pred = sill * (1.0 - np.exp(-emp["h_m"].to_numpy() / theta))
        rmse = float(np.sqrt(np.average((emp["gamma"].to_numpy() - pred) ** 2, weights=emp["n_pairs"].to_numpy())))
        if best is None or rmse < best["rmse"]:
            best = {"model_type": "exponential", "nugget": 0.0, "sill": sill, "theta_z_m": float(theta), "n_pairs": int(emp["n_pairs"].sum()), "rmse": rmse}
    return best


def build_correlation_matrix(z: np.ndarray, theta_z_m: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    h = np.abs(z[:, None] - z[None, :])
    return np.exp(-h / max(theta_z_m, 1e-6))

