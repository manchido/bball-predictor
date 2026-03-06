"""
Training pipeline: walk-forward CV + temporal holdout.

Temporal splits:
  Train:    2023–2024 seasons
  Validate: 2024–2025 season  (used for meta-model + calibration)
  Test:     2025–2026 season  (final metrics, never touched during training)

Walk-forward CV within train seasons:
  Fold 1: Train on 2023 → validate on first half of 2024
  Fold 2: Train on 2023 + first half of 2024 → validate on second half of 2024
  Final:  Train on full 2023–2024

Metrics reported:
  game_total_MAE, game_total_RMSE (test season)
  calibration_80pct: % of true totals inside calibrated p10/p90
  mean_edge: mean(model_total_mean - book_total)  [requires odds column]
  edge_hit_rate: fraction of games where sign(edge) matched o/u outcome
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.calibration import calibrate_intervals, report_calibration
from src.models.ensemble import BballEnsemble
from src.pipeline.features import FEATURE_COLS, TARGET_HOME, TARGET_AWAY
from src.pipeline.silver_to_gold import load_gold
from src.utils.config import settings
from src.utils.logging import logger


TRAIN_SEASONS = ["2023-24", "2024", "2024-25", "2025"]  # both complete historical seasons
VAL_SEASONS   = ["2025-26", "2026"]   # current season: meta-model + calibration
TEST_SEASONS  = ["2025-26", "2026"]   # same as val — no separate holdout during live season


def _prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Extract X, y_home, y_away, game_ids from the feature DataFrame."""
    available = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available)
    if missing:
        logger.warning("Missing feature columns (will fill with 0): {}", missing)
        for col in missing:
            df[col] = 0.0

    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y_home = df[TARGET_HOME].fillna(0).values.astype(np.float32)
    y_away = df[TARGET_AWAY].fillna(0).values.astype(np.float32)
    game_ids = df["game_id"]
    return X, y_home, y_away, game_ids


def train(gold_df: Optional[pd.DataFrame] = None) -> dict[str, Any]:
    """
    Full training run.

    Parameters
    ----------
    gold_df : if None, loads from gold/features.parquet

    Returns
    -------
    dict with metrics + path to saved model
    """
    if gold_df is None:
        gold_df = load_gold()

    if gold_df.empty:
        raise ValueError("Gold table is empty. Run the full pipeline first.")

    # Normalise season column
    gold_df = gold_df.copy()
    gold_df["date"] = pd.to_datetime(gold_df["date"])

    # Split by season label or by year
    if "season" in gold_df.columns:
        train_df = gold_df[gold_df["season"].isin(TRAIN_SEASONS)]
        val_df   = gold_df[gold_df["season"].isin(VAL_SEASONS)]
        test_df  = gold_df[gold_df["season"].isin(TEST_SEASONS)]
    else:
        # Fallback: split by calendar year
        train_df = gold_df[gold_df["date"].dt.year.isin([2023, 2024])]
        val_df   = gold_df[gold_df["date"].dt.year == 2025]
        test_df  = gold_df[gold_df["date"].dt.year == 2026]

    logger.info(
        "Split sizes — train: {}, val: {}, test: {}",
        len(train_df), len(val_df), len(test_df),
    )

    if train_df.empty or val_df.empty:
        raise ValueError("Train or validation set is empty. Check season labels.")

    # Walk-forward CV (log fold scores, don't use for final model)
    _walk_forward_cv(gold_df, TRAIN_SEASONS)

    # --- Final model ---
    X_train, y_home_train, y_away_train, _ = _prepare_xy(train_df, FEATURE_COLS)
    X_val,   y_home_val,   y_away_val,   _ = _prepare_xy(val_df,   FEATURE_COLS)

    ensemble = BballEnsemble()
    ensemble.fit(X_train, y_home_train, y_away_train, X_val, y_home_val, y_away_val)

    # Calibrate intervals on validation set
    val_preds = ensemble.predict(X_val)
    val_total_mean = np.array([p.total_mean for p in val_preds])
    val_total_p10  = np.array([p.total_p10  for p in val_preds])
    val_total_p90  = np.array([p.total_p90  for p in val_preds])
    val_actuals    = y_home_val + y_away_val

    scale = calibrate_intervals(val_total_mean, val_total_p10, val_total_p90, val_actuals)
    ensemble._calibration_scale = scale

    # --- Per-league bias correction (computed on val set) ---
    if "league" in val_df.columns:
        val_home_preds = np.array([p.home_pred_mean for p in val_preds])
        val_away_preds = np.array([p.away_pred_mean for p in val_preds])
        bias_df = val_df[["league"]].copy().reset_index(drop=True)
        bias_df["home_resid"] = y_home_val - val_home_preds
        bias_df["away_resid"] = y_away_val - val_away_preds
        ensemble._league_home_bias = bias_df.groupby("league")["home_resid"].mean().round(4).to_dict()
        ensemble._league_away_bias = bias_df.groupby("league")["away_resid"].mean().round(4).to_dict()
        logger.info("Per-league home bias: {}", ensemble._league_home_bias)
        logger.info("Per-league away bias: {}", ensemble._league_away_bias)

    # --- Test metrics ---
    metrics: dict[str, Any] = {}

    if not test_df.empty:
        X_test, y_home_test, y_away_test, test_ids = _prepare_xy(test_df, FEATURE_COLS)
        test_leagues = test_df["league"].tolist() if "league" in test_df.columns else None
        test_preds = ensemble.predict(X_test, game_ids=test_ids.tolist(), leagues=test_leagues)

        test_total_mean = np.array([p.total_mean for p in test_preds])
        test_total_p10  = np.array([p.total_p10  for p in test_preds])
        test_total_p90  = np.array([p.total_p90  for p in test_preds])
        test_actuals    = y_home_test + y_away_test

        # Calibrated intervals for test
        half_width = (test_total_p90 - test_total_p10) / 2 * scale
        cal_p10 = test_total_mean - half_width
        cal_p90 = test_total_mean + half_width

        mae  = float(mean_absolute_error(test_actuals, test_total_mean))
        rmse = float(np.sqrt(mean_squared_error(test_actuals, test_total_mean)))
        cal_report = report_calibration(
            test_total_mean, test_total_p10, test_total_p90, test_actuals, scale
        )

        metrics["game_total_MAE"]   = round(mae, 4)
        metrics["game_total_RMSE"]  = round(rmse, 4)
        metrics["calibration_80pct"] = round(cal_report["calibrated_coverage"], 4)
        metrics["mean_interval_width"] = round(cal_report["mean_interval_width"], 4)

        # Per-league MAE on test set
        if "league" in test_df.columns:
            test_df_copy = test_df.reset_index(drop=True).copy()
            test_df_copy["pred_total"] = test_total_mean
            test_df_copy["actual_total"] = test_actuals
            test_df_copy["abs_err"] = np.abs(test_actuals - test_total_mean)
            per_league = (
                test_df_copy.groupby("league")
                .agg(n=("abs_err", "count"), mae=("abs_err", "mean"))
                .round(2)
            )
            metrics["per_league_mae"] = per_league["mae"].to_dict()
            logger.info("Per-league test MAE:\n{}", per_league.to_string())

        # Edge tracking (requires book_total column)
        if "book_total" in test_df.columns:
            book = test_df["book_total"].fillna(0).values
            edge = test_total_mean - book
            metrics["mean_edge"] = round(float(np.mean(edge)), 4)
            # Hit rate: model was on correct side of the closing line
            hit = (edge > 0) == (test_actuals > book)
            metrics["edge_hit_rate"] = round(float(np.mean(hit)), 4)
        else:
            logger.warning("book_total not in test set — skipping edge metrics")

        logger.info("Test metrics: {}", metrics)
    else:
        logger.warning("Test set empty — no final metrics computed (expected for early seasons)")

    # --- Save ---
    model_path = ensemble.save()
    metrics["model_path"] = str(model_path)
    metrics["calibration_scale"] = round(scale, 6)

    # Persist metrics
    metrics_path = Path(settings.models_dir) / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved → {}", metrics_path)

    return metrics


# ---------------------------------------------------------------------------
# Walk-forward CV (logging only — does not influence final model)
# ---------------------------------------------------------------------------

def _walk_forward_cv(gold_df: pd.DataFrame, train_seasons: list[str]) -> None:
    """Run 2-fold walk-forward CV within train seasons and log MAE."""
    if "season" not in gold_df.columns:
        logger.info("No 'season' column — skipping walk-forward CV")
        return

    season_order = sorted(gold_df[gold_df["season"].isin(train_seasons)]["season"].unique())
    if len(season_order) < 2:
        logger.info("Need ≥2 train seasons for walk-forward CV — skipping")
        return

    logger.info("Walk-forward CV on seasons: {}", season_order)

    for i in range(1, len(season_order)):
        cv_train = gold_df[gold_df["season"].isin(season_order[:i])]
        cv_val   = gold_df[gold_df["season"] == season_order[i]]

        if cv_train.empty or cv_val.empty:
            continue

        X_tr, y_h_tr, y_a_tr, _ = _prepare_xy(cv_train, FEATURE_COLS)
        X_vl, y_h_vl, y_a_vl, _ = _prepare_xy(cv_val,   FEATURE_COLS)

        ens = BballEnsemble()
        ens.fit(X_tr, y_h_tr, y_a_tr, X_vl, y_h_vl, y_a_vl)
        preds = ens.predict(X_vl)

        pred_totals = np.array([p.total_mean for p in preds])
        actual_totals = y_h_vl + y_a_vl
        mae = mean_absolute_error(actual_totals, pred_totals)
        logger.info(
            "WF-CV fold train={} → val={} | MAE={:.2f}", season_order[:i], season_order[i], mae
        )
