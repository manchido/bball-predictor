"""
Interval calibration.

Goal: p10/p90 prediction interval should contain the true game total ~80%
of the time (conformal-style empirical calibration on the validation season).

Method:
  1. Predict on validation set → raw p10/p90 for each game
  2. Compute empirical coverage:
       coverage = mean(p10 <= actual <= p90)
  3. If coverage != 80%, solve for a scale_factor s.t.:
       adjusted_p10 = mean - (mean - raw_p10) * scale_factor
       adjusted_p90 = mean + (raw_p90 - mean) * scale_factor
     That gives coverage ≈ 80%.
  4. Store scale_factor on the BballEnsemble object.
"""

from __future__ import annotations

import numpy as np
from src.utils.logging import logger


def compute_coverage(
    actuals: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
) -> float:
    """Fraction of actuals contained within [p10, p90]."""
    return float(np.mean((actuals >= p10) & (actuals <= p90)))


def calibrate_intervals(
    total_mean: np.ndarray,
    total_p10: np.ndarray,
    total_p90: np.ndarray,
    actuals: np.ndarray,
    target_coverage: float = 0.80,
    tol: float = 0.005,
    max_iter: int = 100,
) -> float:
    """
    Binary search for scale_factor that achieves target_coverage.

    Returns the scale_factor to be stored on the ensemble.
    scale_factor > 1 widens intervals, < 1 narrows them.
    """
    raw_coverage = compute_coverage(actuals, total_p10, total_p90)
    logger.info(
        "Raw interval coverage on validation set: {:.1%} (target {:.1%})",
        raw_coverage, target_coverage,
    )

    lo, hi = 0.1, 10.0

    for _ in range(max_iter):
        scale = (lo + hi) / 2
        half_width = (total_p90 - total_p10) / 2 * scale
        adj_p10 = total_mean - half_width
        adj_p90 = total_mean + half_width
        cov = compute_coverage(actuals, adj_p10, adj_p90)

        if abs(cov - target_coverage) < tol:
            break
        if cov < target_coverage:
            lo = scale
        else:
            hi = scale

    logger.info(
        "Calibrated scale_factor={:.4f} → coverage={:.1%}", scale, cov
    )
    return float(scale)


def report_calibration(
    total_mean: np.ndarray,
    total_p10: np.ndarray,
    total_p90: np.ndarray,
    actuals: np.ndarray,
    scale_factor: float,
) -> dict[str, float]:
    """Return a calibration summary dict for logging/reporting."""
    half_width = (total_p90 - total_p10) / 2 * scale_factor
    adj_p10 = total_mean - half_width
    adj_p90 = total_mean + half_width

    return {
        "raw_coverage": compute_coverage(actuals, total_p10, total_p90),
        "calibrated_coverage": compute_coverage(actuals, adj_p10, adj_p90),
        "scale_factor": scale_factor,
        "mean_interval_width": float(np.mean(adj_p90 - adj_p10)),
    }
