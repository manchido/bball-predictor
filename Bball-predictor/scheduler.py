"""
APScheduler daily pipeline.

Runs at 09:00 America/St_Johns every day:
  1. Scrape today's injury reports
  2. Fetch bookmaker odds
  3. Scrape today's schedule
  4. Build features
  5. Predict
  6. Compute edge
  7. Append to tracking/daily_tracker.csv

The scheduler is started by the FastAPI lifespan on app startup.
It can also be run standalone for testing.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.utils.config import settings
from src.utils.logging import logger, setup_logging

_TZ = ZoneInfo(settings.timezone)
_scheduler: AsyncIOScheduler | None = None

LEAGUES = [
    "euroleague", "eurocup", "acb", "bsl", "bbl",
    "nba", "lkl", "koris", "nbl_cz", "aba", "cba", "hung",
]


async def run_daily_pipeline() -> None:
    """Execute the full daily prediction pipeline for all leagues."""
    logger.info("Daily pipeline starting at {}", datetime.now(_TZ).isoformat())

    from src.scrapers.injuries import InjuryScraper
    from src.scrapers.odds import OddsClient
    from src.scrapers.live_schedule import LiveScheduleFetcher
    from src.pipeline.silver_to_gold import load_gold
    from src.injury.adjuster import InjuryAdjuster
    from src.models.ensemble import BballEnsemble
    from src.api.routers.predict import _predict_games

    # Load model
    try:
        ensemble = BballEnsemble.load()
    except FileNotFoundError:
        logger.error("No trained model found — skipping daily pipeline. Run 'train' first.")
        return

    scraper = LiveScheduleFetcher()
    injury_scraper = InjuryScraper()
    odds_client = OddsClient()
    gold_df = load_gold()
    adjuster = InjuryAdjuster.from_silver()
    now = datetime.now(_TZ)

    for league in LEAGUES:
        try:
            games = await scraper.fetch_today(league)
            if not games:
                logger.info("No games today: {}", league)
                continue

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

            logger.info(
                "Daily pipeline: {} — {} predictions generated",
                league, len(predictions),
            )

        except Exception as exc:
            logger.exception("Daily pipeline error for league={}", league)

    logger.info("Daily pipeline complete")


def start_scheduler() -> None:
    """Start the APScheduler. Called by FastAPI lifespan."""
    global _scheduler
    if _scheduler and _scheduler.running:
        logger.info("Scheduler already running")
        return

    _scheduler = AsyncIOScheduler(timezone=str(_TZ))
    _scheduler.add_job(
        run_daily_pipeline,
        trigger=CronTrigger(hour=9, minute=0, timezone=str(_TZ)),
        id="daily_pipeline",
        name="Daily prediction pipeline",
        replace_existing=True,
        misfire_grace_time=3600,  # Allow up to 1h late start
    )
    _scheduler.start()
    logger.info(
        "Scheduler started — daily pipeline at 09:00 {}",
        settings.timezone,
    )


def stop_scheduler() -> None:
    """Stop the scheduler gracefully."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")


# ---------------------------------------------------------------------------
# Standalone entry point (for testing the pipeline manually)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging(log_level="DEBUG")
    asyncio.run(run_daily_pipeline())
