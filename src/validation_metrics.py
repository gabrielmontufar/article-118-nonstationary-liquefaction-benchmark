"""Validation metrics used by the site-calibrated application."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

NORM = NormalDist()


def auc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    s = pd.Series(scores)
    y = pd.Series(labels).astype(int)
    valid = s.notna() & y.notna()
    s = s[valid]
    y = y[valid]
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = s.rank(method="average")
    rank_sum_pos = float(ranks[y.eq(1)].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mu = x.mean(axis=0)
    sd = np.maximum(x.std(axis=0), 1e-9)
    z = (x - mu) / sd
    X = np.column_stack([np.ones(len(z)), z])
    beta = np.zeros(X.shape[1])
    for _ in range(80):
        eta = np.clip(X @ beta, -35, 35)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-8, None)
        penalty = np.r_[0.0, beta[1:]] * l2
        grad = X.T @ (y - p) - penalty
        hess = -(X.T * w) @ X
        ridge = np.diag(np.r_[0.0, np.ones(X.shape[1] - 1) * l2])
        step = np.linalg.solve(hess - ridge - np.eye(X.shape[1]) * 1e-6, grad)
        beta -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return np.r_[beta, mu, sd]


def predict_logistic(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n_feat = x.shape[1]
    beta = params[: n_feat + 1]
    mu = params[n_feat + 1 : 2 * n_feat + 1]
    sd = np.maximum(params[2 * n_feat + 1 :], 1e-9)
    z = (x - mu) / sd
    eta = np.clip(np.column_stack([np.ones(len(z)), z]) @ beta, -35, 35)
    return 1.0 / (1.0 + np.exp(-eta))


def calibration_intercept_slope(prob: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))[:, None]
    params = fit_logistic(logit, np.asarray(y, dtype=float))
    return float(params[0]), float(params[1])


def binary_metrics(prob: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    cal_i, cal_s = calibration_intercept_slope(p, y)
    return {
        "auc": auc_rank(p, y),
        "brier_score": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "calibration_intercept": cal_i,
        "calibration_slope": cal_s,
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
