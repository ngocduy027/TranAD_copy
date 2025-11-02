import os
from typing import Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def compute_roc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve and AUC.

    Parameters
    - y_true: binary ground-truth array shaped (T, F) or (T,) with values in {0,1}
    - y_score: anomaly scores shaped like y_true

    Returns
    - fpr, tpr, thresholds, auc
    """
    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score).astype(float)
    if y_true.shape != y_score.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}")

    # Flatten for micro-average
    y_true_flat = y_true.ravel()
    y_score_flat = y_score.ravel()

    # Drop NaNs if any
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_score_flat))
    y_true_flat = y_true_flat[mask]
    y_score_flat = y_score_flat[mask]

    if y_true_flat.size == 0:
        raise ValueError("Empty inputs after NaN filtering.")

    # Guard against constant labels
    if np.all(y_true_flat == 0) or np.all(y_true_flat == 1):
        # AUC undefined if only one class present
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        thresholds = np.array([np.inf, -np.inf])
        return fpr, tpr, thresholds, float("nan")

    fpr, tpr, thresholds = roc_curve(y_true_flat, y_score_flat)
    auc = roc_auc_score(y_true_flat, y_score_flat)
    return fpr, tpr, thresholds, auc


def plot_roc(
    scores: np.ndarray,
    labels: np.ndarray,
    out_file: str = os.path.join("plots", "roc_micro.png"),
    title: Optional[str] = None,
    per_dim: Optional[int] = None,
) -> Dict[str, float]:
    """
    Plot ROC curve and save to out_file.

    - If per_dim is None, computes micro-average over all dimensions (flattened).
    - If per_dim is an integer, plots ROC for that feature index using scores[:, per_dim].

    Returns
    - dict with keys: 'auc', 'points' (number of evaluated points)
    """
    scores = np.asarray(scores).astype(float)
    labels = np.asarray(labels).astype(float)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    if per_dim is not None:
        if per_dim < 0 or per_dim >= scores.shape[1]:
            raise IndexError(f"per_dim={per_dim} out of bounds for features={scores.shape[1]}")
        y_score = scores[:, per_dim]
        y_true = labels[:, per_dim]
        fpr, tpr, thresholds, auc = compute_roc(y_true, y_score)
        plot_title = title or f"ROC (dim {per_dim}) AUC={auc:.4f}" if not np.isnan(auc) else f"ROC (dim {per_dim})"
    else:
        fpr, tpr, thresholds, auc = compute_roc(labels, scores)
        plot_title = title or (f"ROC (micro) AUC={auc:.4f}" if not np.isnan(auc) else "ROC (micro)")

    _ensure_dir(out_file)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}" if not np.isnan(auc) else "AUC=N/A")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(plot_title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()

    return {"auc": float(auc) if not np.isnan(auc) else float("nan"), "points": int(scores.size)}


# Convenience: plot per-dimension ROC curves to a folder
def plot_roc_per_dim(
    scores: np.ndarray,
    labels: np.ndarray,
    out_dir: str = os.path.join("plots", "roc_per_dim"),
) -> None:
    """
    Plot ROC curves for each feature dimension into out_dir.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    _ensure_dir(os.path.join(out_dir, "dummy"))
    for i in range(scores.shape[1]):
        out_file = os.path.join(out_dir, f"roc_dim_{i}.png")
        try:
            plot_roc(scores, labels, out_file=out_file, per_dim=i)
        except Exception:
            # Continue on per-dimension failures
            continue

