"""POST /backtest — run backtest over a date range."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.api.schemas import BacktestRequest, BacktestMetrics
from src.models.calibration import compute_coverage
from src.pipeline.features import FEATURE_COLS
from src.pipeline.silver_to_gold import load_gold
from src.utils.logging import logger

router = APIRouter(tags=["backtest"])


@router.post("", response_model=BacktestMetrics)
async def run_backtest(request: BacktestRequest) -> BacktestMetrics:
    """
    Backtest the current model on historical games in [start_date, end_date].

    Uses the gold table (pre-computed features + actuals) — no re-scraping.
    """
    from src.api.routers.predict import _get_ensemble
    ensemble = _get_ensemble()

    gold_df = load_gold()
    if gold_df.empty:
        raise HTTPException(status_code=503, detail="Gold table empty — run training pipeline first")

    gold_df["date"] = pd.to_datetime(gold_df["date"])
    mask = (
        (gold_df["date"].dt.date >= request.start_date)
        & (gold_df["date"].dt.date <= request.end_date)
    )
    if request.leagues:
        if "league" in gold_df.columns:
            mask &= gold_df["league"].isin(request.leagues)

    subset = gold_df[mask].dropna(subset=["home_points", "away_points"])

    if subset.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data in range {request.start_date} – {request.end_date}",
        )

    available = [c for c in FEATURE_COLS if c in subset.columns]
    X = subset[FEATURE_COLS].fillna(0).values.astype(np.float32)
    actuals = (subset["home_points"] + subset["away_points"]).values

    preds = ensemble.predict(X, game_ids=subset["game_id"].tolist())
    pred_means = np.array([p.total_mean for p in preds])
    pred_p10   = np.array([p.total_p10  for p in preds])
    pred_p90   = np.array([p.total_p90  for p in preds])

    # Apply calibration scale
    scale = ensemble._calibration_scale
    half_width = (pred_p90 - pred_p10) / 2 * scale
    cal_p10 = pred_means - half_width
    cal_p90 = pred_means + half_width

    mae  = float(mean_absolute_error(actuals, pred_means))
    rmse = float(np.sqrt(mean_squared_error(actuals, pred_means)))
    cal  = float(compute_coverage(actuals, cal_p10, cal_p90))
    width = float(np.mean(cal_p90 - cal_p10))

    mean_edge: float | None = None
    hit_rate: float | None = None
    if "book_total" in subset.columns:
        book = subset["book_total"].fillna(0).values
        valid = book > 0
        if valid.any():
            edge = pred_means[valid] - book[valid]
            mean_edge = float(np.mean(edge))
            hits = (edge > 0) == (actuals[valid] > book[valid])
            hit_rate = float(np.mean(hits))

    leagues_used = subset["league"].unique().tolist() if "league" in subset.columns else []

    logger.info(
        "Backtest {}/{}: n={} MAE={:.2f} RMSE={:.2f} cal={:.1%}",
        request.start_date, request.end_date, len(subset), mae, rmse, cal,
    )

    return BacktestMetrics(
        start_date=request.start_date,
        end_date=request.end_date,
        n_games=len(subset),
        game_total_mae=round(mae, 4),
        game_total_rmse=round(rmse, 4),
        calibration_80pct=round(cal, 4),
        mean_interval_width=round(width, 4),
        mean_edge=round(mean_edge, 4) if mean_edge is not None else None,
        edge_hit_rate=round(hit_rate, 4) if hit_rate is not None else None,
        leagues=leagues_used,
    )
