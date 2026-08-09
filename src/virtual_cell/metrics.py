from __future__ import annotations

import numpy as np


def _paired_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any():
        raise ValueError("No paired finite observations available")
    return y_true, y_pred, valid


def log2_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred, valid = _paired_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(y_true[valid] - y_pred[valid]))))


def global_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred, valid = _paired_arrays(y_true, y_pred)
    truth = y_true[valid]
    prediction = y_pred[valid]
    denominator = np.square(truth - truth.mean()).sum()
    if denominator <= 0:
        return float("nan")
    return float(1.0 - np.square(truth - prediction).sum() / denominator)


def protein_r2_median(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int]:
    y_true, y_pred, _ = _paired_arrays(y_true, y_pred)
    scores: list[float] = []
    for column in range(y_true.shape[1]):
        valid = np.isfinite(y_true[:, column]) & np.isfinite(y_pred[:, column])
        if valid.sum() < 2:
            continue
        truth = y_true[valid, column]
        denominator = np.square(truth - truth.mean()).sum()
        if denominator <= 0:
            continue
        prediction = y_pred[valid, column]
        score = 1.0 - np.square(truth - prediction).sum() / denominator
        if np.isfinite(score):
            scores.append(float(score))
    if not scores:
        return float("nan"), 0
    return float(np.median(np.asarray(scores))), len(scores)


def diagnostic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    median_r2, evaluated_proteins = protein_r2_median(y_true, y_pred)
    paired = np.isfinite(y_true) & np.isfinite(y_pred)
    return {
        "log2_rmse": log2_rmse(y_true, y_pred),
        "global_r2": global_r2(y_true, y_pred),
        "protein_r2_median": median_r2,
        "evaluated_proteins": evaluated_proteins,
        "paired_values": int(paired.sum()),
    }

