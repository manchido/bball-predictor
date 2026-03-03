"""GET /tracking — query the daily tracker CSV."""

from __future__ import annotations

import csv
from datetime import date
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import TrackerRow, TrackingResponse
from src.utils.config import settings
from src.utils.logging import logger

router = APIRouter(tags=["tracking"])


@router.get("", response_model=TrackingResponse)
async def get_tracking(
    date_from: Optional[date] = Query(default=None),
    date_to: Optional[date] = Query(default=None),
    league: Optional[str] = Query(default=None),
    limit: int = Query(default=200, le=1000),
) -> TrackingResponse:
    """
    Return rows from daily_tracker.csv.

    Supports date range and league filtering.
    Columns: date, league, match, book_total, model_total_mean, edge, actual_total, error, odds_source
    """
    tracker_path = settings.tracking_path
    if not tracker_path.exists():
        return TrackingResponse(rows=[], total=0, date_from=date_from, date_to=date_to)

    rows: list[TrackerRow] = []
    with open(tracker_path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            try:
                row_date = date.fromisoformat(raw["date"])
                if date_from and row_date < date_from:
                    continue
                if date_to and row_date > date_to:
                    continue
                if league and raw.get("league", "").lower() != league.lower():
                    continue

                rows.append(TrackerRow(
                    date=row_date,
                    league=raw.get("league", ""),
                    match=raw.get("match", ""),
                    book_total=float(raw["book_total"]) if raw.get("book_total") else None,
                    book_spread=float(raw["book_spread"]) if raw.get("book_spread") else None,
                    model_total_mean=float(raw.get("model_total_mean", 0)),
                    model_total_p10=float(raw.get("model_total_p10", 0)),
                    model_total_p90=float(raw.get("model_total_p90", 0)),
                    confidence=float(raw.get("confidence", 0)),
                    edge=float(raw["edge"]) if raw.get("edge") else None,
                    actual_total=float(raw["actual_total"]) if raw.get("actual_total") else None,
                    error=float(raw["error"]) if raw.get("error") else None,
                    odds_source=raw.get("odds_source") or None,
                ))
            except (ValueError, KeyError) as exc:
                logger.debug("Tracker row parse error: {}", exc)

    total = len(rows)
    rows = rows[-limit:]  # return most recent N

    return TrackingResponse(
        rows=rows,
        total=total,
        date_from=date_from,
        date_to=date_to,
    )


@router.post("/update_actuals")
async def update_actuals() -> dict:
    """
    Scan tracker for rows missing actual_total and attempt to fill
    from silver box score data for completed games.
    Returns count of rows updated.
    """
    tracker_path = settings.tracking_path
    if not tracker_path.exists():
        raise HTTPException(status_code=404, detail="Tracker not found")

    try:
        from src.pipeline.bronze_to_silver import load_silver
        bs_df = load_silver("box_scores")
        if bs_df.empty:
            return {"updated": 0, "message": "No silver box score data"}

        # game_id → actual_total mapping
        if "game_id" in bs_df.columns and "points" in bs_df.columns:
            actuals_map = (
                bs_df.groupby("game_id")["points"].sum().to_dict()
            )
        else:
            return {"updated": 0, "message": "Missing columns in silver box scores"}

        df = pd.read_csv(tracker_path)
        updated = 0
        for idx, row in df.iterrows():
            if pd.isna(row.get("actual_total")) or row.get("actual_total") == "":
                game_id = row.get("game_id", "")
                if game_id in actuals_map:
                    actual = actuals_map[game_id]
                    df.at[idx, "actual_total"] = actual
                    df.at[idx, "error"] = actual - float(row.get("model_total_mean", 0))
                    updated += 1

        df.to_csv(tracker_path, index=False)
        logger.info("Updated {} rows in tracker with actuals", updated)
        return {"updated": updated}

    except Exception as exc:
        logger.exception("Failed to update actuals")
        raise HTTPException(status_code=500, detail=str(exc))
