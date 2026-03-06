"""
GET /predict/today       — today's game predictions across all leagues
GET /predict/{game_id}   — fetch a stored prediction by game_id
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import PredictionResponse, TodayPredictionsResponse
from src.scrapers.injuries import InjuryScraper
from src.scrapers.live_schedule import LiveScheduleFetcher
from src.scrapers.odds import OddsClient
from src.pipeline.features import FEATURE_COLS, build_matchup_features
from src.pipeline.silver_to_gold import load_gold
from src.injury.adjuster import InjuryAdjuster
from src.utils.config import settings
from src.utils.logging import logger

_TZ = ZoneInfo(settings.timezone)

router = APIRouter(tags=["predictions"])

LEAGUES = [
    "euroleague", "eurocup", "acb", "bsl", "bbl",
    "nba", "lkl", "koris", "nbl_cz", "aba", "cba", "hung",
]

# Module-level model handle (loaded at app startup via lifespan)
_ensemble = None


def set_ensemble(model) -> None:
    global _ensemble
    _ensemble = model


def _get_ensemble():
    if _ensemble is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run POST /train first.",
        )
    return _ensemble


_LEAGUE_TIMEOUT = 20  # seconds per league before giving up


async def _process_league(
    league: str,
    scraper: LiveScheduleFetcher,
    injury_scraper: InjuryScraper,
    odds_client: OddsClient,
    ensemble,
    gold_df,
    adjuster: InjuryAdjuster,
    now: datetime,
) -> tuple[list[PredictionResponse], list[str]]:
    """Fetch + predict for one league. Returns (predictions, warnings)."""
    try:
        games = await scraper.fetch_today(league)
        if not games:
            logger.info("No games today for league={}", league)
            return [], []

        injury_data, odds_records = await asyncio.gather(
            injury_scraper.fetch_today(league),
            odds_client.fetch_today_odds(league),
        )

        predictions = _predict_games(
            ensemble=ensemble,
            games=games,
            gold_df=gold_df,
            odds_records=odds_records,
            injury_data=injury_data,
            adjuster=adjuster,
            league=league,
            now=now,
        )
        return predictions, []

    except Exception as exc:
        logger.warning("Prediction skipped for league={}: {}", league, exc)
        return [], [f"{league}: {type(exc).__name__}"]


@router.get("/today", response_model=TodayPredictionsResponse)
async def predict_today(
    leagues: Optional[list[str]] = Query(default=None),
) -> TodayPredictionsResponse:
    """
    Full daily prediction pipeline — all leagues run concurrently.
    Each league is capped at 20 s; failures are skipped with a warning.
    """
    ensemble = _get_ensemble()
    active_leagues = leagues or LEAGUES
    today = date.today()
    now = datetime.now(_TZ)

    scraper       = LiveScheduleFetcher()
    injury_scraper = InjuryScraper()
    odds_client   = OddsClient()
    gold_df       = load_gold()
    adjuster      = InjuryAdjuster.from_silver()  # load once, share across leagues

    if gold_df.empty:
        return TodayPredictionsResponse(
            date=today, leagues=active_leagues, games=[],
            model_loaded=True, warnings=["Gold table empty — run build-gold first"],
        )

    async def _with_timeout(league: str):
        try:
            return await asyncio.wait_for(
                _process_league(league, scraper, injury_scraper, odds_client,
                                ensemble, gold_df, adjuster, now),
                timeout=_LEAGUE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("League {} timed out after {}s", league, _LEAGUE_TIMEOUT)
            return [], [f"{league}: timed out"]

    results = await asyncio.gather(*[_with_timeout(lg) for lg in active_leagues])

    all_predictions: list[PredictionResponse] = []
    all_warnings: list[str] = []
    for preds, warns in results:
        all_predictions.extend(preds)
        all_warnings.extend(warns)

    return TodayPredictionsResponse(
        date=today,
        leagues=active_leagues,
        games=all_predictions,
        model_loaded=True,
        warnings=all_warnings,
    )


@router.get("/{game_id}", response_model=PredictionResponse)
async def get_prediction(game_id: str) -> PredictionResponse:
    """Retrieve a stored prediction from the daily tracker by game_id."""
    import csv
    tracker = settings.tracking_path
    if not tracker.exists():
        raise HTTPException(status_code=404, detail="Tracker not found")

    with open(tracker, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("game_id") == game_id:
                return _tracker_row_to_prediction(row)

    raise HTTPException(status_code=404, detail=f"game_id {game_id!r} not found in tracker")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _predict_games(
    ensemble,
    games: list[dict],
    gold_df: pd.DataFrame,
    odds_records: dict,
    injury_data: dict,
    adjuster: InjuryAdjuster,
    league: str,
    now: datetime,
) -> list[PredictionResponse]:
    """Build features + predict for a list of today's games."""
    predictions: list[PredictionResponse] = []

    for game in games:
        game_id = game["game_id"]
        home_team_id = game["home_team_id"]
        away_team_id = game["away_team_id"]

        # Extract rolling context from gold table for home + away team
        home_ctx = gold_df[gold_df["home_team_id"] == home_team_id].tail(10)
        away_ctx = gold_df[gold_df["away_team_id"] == away_team_id].tail(10)

        if home_ctx.empty or away_ctx.empty:
            logger.debug("Insufficient historical data for game_id={}", game_id)
            continue

        # Use most recent row stats as proxy for current rolling values
        feature_row = {}
        for col in FEATURE_COLS:
            if col in home_ctx.columns:
                feature_row[col] = home_ctx.iloc[-1][col]
            else:
                feature_row[col] = 0.0

        # Override away features
        for col in FEATURE_COLS:
            if col.startswith("away_") and col in away_ctx.columns:
                feature_row[col] = away_ctx.iloc[-1][col]

        feature_row["home_team_id"] = home_team_id
        feature_row["away_team_id"] = away_team_id

        # Apply injury adjustment
        current_season = _infer_season(now)
        feature_row = adjuster.apply_adjustment(feature_row, injury_data, season=current_season)

        # Predict (pass league for per-league bias correction)
        X = np.array([[feature_row.get(c, 0.0) for c in FEATURE_COLS]], dtype=np.float32)
        preds = ensemble.predict(X, game_ids=[game_id], leagues=[league])
        p = preds[0]

        # Odds
        odds = odds_records.get(game_id)
        book_total = odds.book_total if odds else None
        book_spread = odds.book_spread if odds else None
        odds_source = odds.odds_source if odds else None
        edge = round(p.total_mean - book_total, 2) if book_total else None

        import math

        def _safe(val, digits=1):
            return round(val, digits) if val and not math.isnan(val) and val > 0 else None

        # Pace (kept in model, passed through for completeness)
        home_pace = _safe(feature_row.get("home_pace_per40_l5"), 2)
        away_pace = _safe(feature_row.get("away_pace_per40_l5"), 2)

        # Rolling scoring form (L5 avg) — what actually drives the total
        home_pts_l5         = _safe(feature_row.get("home_points_l5"))
        away_pts_l5         = _safe(feature_row.get("away_points_l5"))
        home_pts_allowed_l5 = _safe(feature_row.get("home_opp_points_l5"))
        away_pts_allowed_l5 = _safe(feature_row.get("away_opp_points_l5"))

        match_str = f"{game['away_team']} @ {game['home_team']}"
        pred_resp = PredictionResponse(
            game_id=game_id,
            league=league,
            match=match_str,
            date=date.fromisoformat(game["date"]),
            book_total=book_total,
            book_spread=book_spread,
            model_total_mean=round(p.total_mean, 2),
            model_total_p10=round(p.total_p10, 2),
            model_total_p50=round(p.total_p50, 2),
            model_total_p90=round(p.total_p90, 2),
            model_home_mean=round(p.home_pred_mean, 2),
            model_away_mean=round(p.away_pred_mean, 2),
            confidence=round(p.confidence, 4),
            edge=edge,
            odds_source=odds_source,
            timestamp=now,
            home_pace=home_pace,
            away_pace=away_pace,
            home_pts_l5=home_pts_l5,
            away_pts_l5=away_pts_l5,
            home_pts_allowed_l5=home_pts_allowed_l5,
            away_pts_allowed_l5=away_pts_allowed_l5,
        )
        predictions.append(pred_resp)

        # Append to tracker
        _append_tracker(pred_resp, game)

    return predictions


def _append_tracker(pred: PredictionResponse, game: dict) -> None:
    """Append prediction to daily_tracker.csv."""
    import csv
    tracker = settings.tracking_path
    tracker.parent.mkdir(parents=True, exist_ok=True)
    write_header = not tracker.exists()

    fields = [
        "date", "league", "match", "game_id",
        "book_total", "book_spread",
        "model_total_mean", "model_total_p10", "model_total_p90",
        "confidence", "edge",
        "actual_total", "error",
        "odds_source", "timestamp",
    ]
    row = {
        "date": pred.date.isoformat(),
        "league": pred.league,
        "match": pred.match,
        "game_id": pred.game_id,
        "book_total": pred.book_total or "",
        "book_spread": pred.book_spread or "",
        "model_total_mean": pred.model_total_mean,
        "model_total_p10": pred.model_total_p10,
        "model_total_p90": pred.model_total_p90,
        "confidence": pred.confidence,
        "edge": pred.edge or "",
        "actual_total": "",   # filled in post-game
        "error": "",          # filled in post-game
        "odds_source": pred.odds_source or "",
        "timestamp": pred.timestamp.isoformat(),
    }

    with open(tracker, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _tracker_row_to_prediction(row: dict) -> PredictionResponse:
    return PredictionResponse(
        game_id=row.get("game_id", ""),
        league=row.get("league", ""),
        match=row.get("match", ""),
        date=date.fromisoformat(row["date"]),
        book_total=float(row["book_total"]) if row.get("book_total") else None,
        book_spread=float(row["book_spread"]) if row.get("book_spread") else None,
        model_total_mean=float(row.get("model_total_mean", 0)),
        model_total_p10=float(row.get("model_total_p10", 0)),
        model_total_p50=float(row.get("model_total_mean", 0)),
        model_total_p90=float(row.get("model_total_p90", 0)),
        model_home_mean=0.0,
        model_away_mean=0.0,
        confidence=float(row.get("confidence", 0)),
        edge=float(row["edge"]) if row.get("edge") else None,
        odds_source=row.get("odds_source"),
        timestamp=datetime.fromisoformat(row["timestamp"]) if row.get("timestamp") else datetime.now(_TZ),
    )


def _infer_season(now: datetime) -> str:
    """Infer current basketball season string from date."""
    year = now.year
    month = now.month
    # European basketball seasons start ~October
    if month >= 10:
        return f"{year}-{str(year + 1)[-2:]}"
    return f"{year - 1}-{str(year)[-2:]}"
