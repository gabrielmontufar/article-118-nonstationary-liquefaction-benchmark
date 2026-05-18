"""Groundwater trajectory calibration for the site-calibrated extension."""

from __future__ import annotations

import numpy as np
import pandas as pd


def decimal_year(dates: pd.Series) -> np.ndarray:
    dt = pd.Series(pd.to_datetime(dates))
    year_start = pd.to_datetime(dt.dt.year.astype(str) + "-01-01")
    next_year = pd.to_datetime((dt.dt.year + 1).astype(str) + "-01-01")
    return dt.dt.year.to_numpy() + ((dt - year_start) / (next_year - year_start)).to_numpy()


def fit_groundwater_model(df: pd.DataFrame, event_date: str, period_years: float = 1.0) -> tuple[dict, pd.DataFrame]:
    data = df.dropna(subset=["date", "gw_depth_m"]).copy()
    data["t_year"] = decimal_year(data["date"])
    event_year = float(decimal_year(pd.Series([event_date]))[0])
    t0 = float(data["t_year"].mean())
    x = data["t_year"].to_numpy() - t0
    omega = 2 * np.pi / period_years
    event_indicator = (data["t_year"].to_numpy() >= event_year).astype(float)
    X = np.column_stack([np.ones(len(data)), x, np.sin(omega * x), np.cos(omega * x), event_indicator])
    y = data["gw_depth_m"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    sigma = float(np.sqrt(np.sum(resid**2) / max(len(y) - X.shape[1], 1)))
    phi = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if len(resid) > 2 and np.std(resid) > 0 else 0.0
    data["gw_depth_fitted_m"] = fitted
    data["gw_depth_ci_low_m"] = fitted - 1.96 * sigma
    data["gw_depth_ci_high_m"] = fitted + 1.96 * sigma
    params = {
        "beta0": float(beta[0]),
        "beta1": float(beta[1]),
        "beta2": float(beta[2]),
        "beta3": float(beta[3]),
        "beta4": float(beta[4]),
        "sigma_residual": sigma,
        "ar1_phi": phi,
        "rmse_m": float(np.sqrt(np.mean(resid**2))),
        "n_obs": int(len(data)),
        "t0_decimal_year": t0,
        "event_decimal_year": event_year,
        "period_years": period_years,
    }
    return params, data


def predict_groundwater(dates: pd.Series, params: dict) -> pd.DataFrame:
    t = decimal_year(dates)
    x = t - params["t0_decimal_year"]
    omega = 2 * np.pi / params["period_years"]
    event_indicator = (t >= params["event_decimal_year"]).astype(float)
    y = (
        params["beta0"]
        + params["beta1"] * x
        + params["beta2"] * np.sin(omega * x)
        + params["beta3"] * np.cos(omega * x)
        + params["beta4"] * event_indicator
    )
    sigma = params["sigma_residual"]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "gw_depth_mean_m": y,
            "gw_depth_ci_low_m": y - 1.96 * sigma,
            "gw_depth_ci_high_m": y + 1.96 * sigma,
        }
    )
