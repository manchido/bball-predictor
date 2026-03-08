"""
GET /predict/today       — today's game predictions across all leagues
GET /predict/{game_id}   — fetch a stored prediction by game_id
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import ManualPredictRequest, PredictionResponse, TodayPredictionsResponse
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
    target_date: date | None = None,
) -> tuple[list[PredictionResponse], list[str]]:
    """Fetch + predict for one league. Returns (predictions, warnings)."""
    try:
        games = await scraper.fetch_today(league, target_date)
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
    date_param: Optional[str] = Query(default=None, alias="date",
                                      description="Date to predict (YYYY-MM-DD / 'today' / 'tomorrow' / 'yesterday')"),
) -> TodayPredictionsResponse:
    """
    Full daily prediction pipeline — all leagues run concurrently.
    Pass ?date=YYYY-MM-DD (or 'today'/'tomorrow'/'yesterday') to predict a specific date.
    Past dates are served from the tracker CSV without re-running the model.
    Each league is capped at 20 s; failures are skipped with a warning.
    """
    # Resolve target date
    today = date.today()
    if date_param is None or date_param.lower() == "today":
        target_date = today
    elif date_param.lower() == "tomorrow":
        target_date = today + timedelta(days=1)
    elif date_param.lower() == "yesterday":
        target_date = today - timedelta(days=1)
    else:
        try:
            target_date = date.fromisoformat(date_param)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date: {date_param!r}. Use YYYY-MM-DD.")

    active_leagues = leagues or LEAGUES

    # Past dates: serve stored predictions from tracker (no model re-run needed)
    if target_date < today:
        return _predictions_from_tracker(target_date, active_leagues)

    # Today or future: live schedule + model
    ensemble = _get_ensemble()
    now = datetime.now(_TZ)

    scraper        = LiveScheduleFetcher()
    injury_scraper = InjuryScraper()
    odds_client    = OddsClient()
    gold_df        = load_gold()
    adjuster       = InjuryAdjuster.from_silver()

    if gold_df.empty:
        return TodayPredictionsResponse(
            date=target_date, leagues=active_leagues, games=[],
            model_loaded=True, warnings=["Gold table empty — run build-gold first"],
        )

    async def _with_timeout(league: str):
        try:
            return await asyncio.wait_for(
                _process_league(league, scraper, injury_scraper, odds_client,
                                ensemble, gold_df, adjuster, now, target_date),
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
        date=target_date,
        leagues=active_leagues,
        games=all_predictions,
        model_loaded=True,
        warnings=all_warnings,
    )


def _predictions_from_tracker(target_date: date, active_leagues: list[str]) -> TodayPredictionsResponse:
    """Return stored predictions for a past date from the tracker CSV."""
    import csv
    tracker = settings.tracking_path
    if not tracker.exists():
        return TodayPredictionsResponse(
            date=target_date, leagues=active_leagues, games=[],
            model_loaded=True, warnings=["No tracker file found. Run predict-today first."],
        )

    league_set = set(active_leagues)
    seen: dict[str, PredictionResponse] = {}
    with open(tracker, newline="") as f:
        for row in csv.DictReader(f):
            try:
                row_date = date.fromisoformat(row["date"])
            except (KeyError, ValueError):
                continue
            if row_date != target_date:
                continue
            if row.get("league") not in league_set:
                continue
            pred = _tracker_row_to_prediction(row)
            # Attach post-game actuals if present
            if row.get("actual_total"):
                try:
                    pred.actual_total = float(row["actual_total"])
                except ValueError:
                    pass
            if row.get("error"):
                try:
                    pred.result_error = float(row["error"])
                except ValueError:
                    pass
            seen[pred.game_id] = pred  # last write wins (dedup)

    return TodayPredictionsResponse(
        date=target_date,
        leagues=active_leagues,
        games=list(seen.values()),
        model_loaded=True,
        warnings=[] if seen else [f"No predictions found for {target_date}."],
    )


@router.post("/game", response_model=PredictionResponse)
async def predict_game(req: ManualPredictRequest) -> PredictionResponse:
    """
    Predict a single manually-specified matchup without needing a live schedule.
    Useful when the schedule scraper is down or for ad-hoc predictions.
    """
    from src.utils.ids import make_game_id, get_team_id

    ensemble = _get_ensemble()
    gold_df = load_gold()
    if gold_df.empty:
        raise HTTPException(status_code=503, detail="Gold table empty — run build-gold first")

    adjuster = InjuryAdjuster.from_silver()
    now = datetime.now(_TZ)
    game_date = req.date or date.today()

    season_year = game_date.year
    season = (
        f"{season_year}-{str(season_year + 1)[-2:]}"
        if game_date.month >= 10
        else f"{season_year - 1}-{str(season_year)[-2:]}"
    )

    game = {
        "game_id":      make_game_id(game_date, req.home_team, req.away_team),
        "date":         game_date.isoformat(),
        "home_team":    req.home_team,
        "away_team":    req.away_team,
        "home_team_id": get_team_id(req.home_team),
        "away_team_id": get_team_id(req.away_team),
        "league":       req.league,
        "season":       season,
        "status":       "manual",
    }

    preds = _predict_games(
        ensemble=ensemble,
        games=[game],
        gold_df=gold_df,
        odds_records={},
        injury_data={},
        adjuster=adjuster,
        league=req.league,
        now=now,
    )

    if not preds:
        raise HTTPException(
            status_code=404,
            detail=f"No historical data found for '{req.home_team}' or '{req.away_team}' in {req.league}. "
                   "Check team names match what's in the gold table.",
        )

    return preds[0]


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

        # Over/Under recommendations — always shown.
        # When a book line exists: compare model vs book line (actionable for betting).
        # When no book line: compare model vs teams' rolling L5 scoring average (trend signal).
        if book_total:
            total_recommendation = "OVER" if p.total_mean > book_total else "UNDER"
            half_line = book_total / 2
            home_recommendation = "OVER" if p.home_pred_mean > half_line else "UNDER"
            away_recommendation = "OVER" if p.away_pred_mean > half_line else "UNDER"
        else:
            # Fallback: compare model prediction vs each team's recent scoring average
            h_ref = feature_row.get("home_points_l5") or 0.0
            a_ref = feature_row.get("away_points_l5") or 0.0
            rolling_total = h_ref + a_ref
            if rolling_total > 0:
                total_recommendation = "OVER" if p.total_mean > rolling_total else "UNDER"
                home_recommendation = "OVER" if p.home_pred_mean > h_ref else "UNDER"
                away_recommendation = "OVER" if p.away_pred_mean > a_ref else "UNDER"
            else:
                total_recommendation = home_recommendation = away_recommendation = None

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
            total_recommendation=total_recommendation,
            home_recommendation=home_recommendation,
            away_recommendation=away_recommendation,
        )
        predictions.append(pred_resp)

        # Append to tracker
        _append_tracker(pred_resp, game)

    return predictions


_TRACKER_FIELDS = [
    "date", "league", "match", "game_id",
    "book_total", "book_spread",
    "model_total_mean", "model_total_p10", "model_total_p90",
    "model_home_mean", "model_away_mean",
    "confidence", "edge",
    "total_recommendation", "home_recommendation", "away_recommendation",
    "actual_total", "error",
    "odds_source", "timestamp",
]


def _ensure_tracker_schema(tracker) -> None:
    """Migrate existing tracker CSV to include any new columns (added in-place)."""
    import csv
    if not tracker.exists():
        return
    with open(tracker, newline="") as f:
        reader = csv.DictReader(f)
        existing_fields = list(reader.fieldnames or [])
        rows = list(reader)
    missing = [c for c in _TRACKER_FIELDS if c not in existing_fields]
    if not missing:
        return
    with open(tracker, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TRACKER_FIELDS)
        writer.writeheader()
        for row in rows:
            for col in missing:
                row.setdefault(col, "")
            writer.writerow({k: row.get(k, "") for k in _TRACKER_FIELDS})


def _append_tracker(pred: PredictionResponse, game: dict) -> None:
    """Append prediction to daily_tracker.csv, skipping if game_id already exists."""
    import csv
    tracker = settings.tracking_path
    tracker.parent.mkdir(parents=True, exist_ok=True)

    _ensure_tracker_schema(tracker)

    # Skip duplicate entries (same game predicted multiple times in one day)
    if tracker.exists():
        with open(tracker, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("game_id") == pred.game_id:
                    return

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
        "model_home_mean": pred.model_home_mean or "",
        "model_away_mean": pred.model_away_mean or "",
        "confidence": pred.confidence,
        "edge": pred.edge or "",
        "total_recommendation": pred.total_recommendation or "",
        "home_recommendation": pred.home_recommendation or "",
        "away_recommendation": pred.away_recommendation or "",
        "actual_total": "",   # filled in post-game
        "error": "",          # filled in post-game
        "odds_source": pred.odds_source or "",
        "timestamp": pred.timestamp.isoformat(),
    }

    write_header = not tracker.exists()
    with open(tracker, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TRACKER_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _tracker_row_to_prediction(row: dict) -> PredictionResponse:
    book_total = float(row["book_total"]) if row.get("book_total") else None
    model_mean = float(row.get("model_total_mean", 0))
    home_mean  = float(row.get("model_home_mean", 0)) if row.get("model_home_mean") else None
    away_mean  = float(row.get("model_away_mean", 0)) if row.get("model_away_mean") else None

    # Restore stored recommendations; recompute if missing (old rows)
    total_rec = row.get("total_recommendation") or None
    home_rec  = row.get("home_recommendation")  or None
    away_rec  = row.get("away_recommendation")  or None
    if not total_rec and book_total and model_mean:
        total_rec = "OVER" if model_mean > book_total else "UNDER"
    if not home_rec and book_total and home_mean:
        half = book_total / 2
        home_rec = "OVER" if home_mean > half else "UNDER"
        away_rec = "OVER" if (away_mean or 0) > half else "UNDER"

    actual = float(row["actual_total"]) if row.get("actual_total") else None
    error  = float(row["error"])        if row.get("error")        else None

    return PredictionResponse(
        game_id=row.get("game_id", ""),
        league=row.get("league", ""),
        match=row.get("match", ""),
        date=date.fromisoformat(row["date"]),
        book_total=book_total,
        book_spread=float(row["book_spread"]) if row.get("book_spread") else None,
        model_total_mean=model_mean,
        model_total_p10=float(row.get("model_total_p10", 0)),
        model_total_p50=model_mean,
        model_total_p90=float(row.get("model_total_p90", 0)),
        model_home_mean=home_mean or 0.0,
        model_away_mean=away_mean or 0.0,
        confidence=float(row.get("confidence", 0)),
        edge=float(row["edge"]) if row.get("edge") else None,
        odds_source=row.get("odds_source") or None,
        timestamp=datetime.fromisoformat(row["timestamp"]) if row.get("timestamp") else datetime.now(_TZ),
        total_recommendation=total_rec,
        home_recommendation=home_rec,
        away_recommendation=away_rec,
        actual_total=actual,
        result_error=error,
    )


def _infer_season(now: datetime) -> str:
    """Infer current basketball season string from date."""
    year = now.year
    month = now.month
    # European basketball seasons start ~October
    if month >= 10:
        return f"{year}-{str(year + 1)[-2:]}"
    return f"{year - 1}-{str(year)[-2:]}"
