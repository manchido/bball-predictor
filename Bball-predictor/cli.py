"""
BballPredictor CLI

Commands:
  train           — run full training pipeline
  predict_today   — predict today's games (prints to stdout)
  backtest        — backtest over a date range
  scrape_test     — smoke-test scrapers without predicting
  build_gold      — (re)build gold feature table from silver data
  update_actuals  — fill in post-game actuals in tracker

Usage:
  python cli.py train
  python cli.py predict_today
  python cli.py backtest --start 2025-10-01 --end 2026-02-28
  python cli.py scrape_test --league euroleague
"""

from __future__ import annotations

import asyncio
import json
from datetime import date
from typing import Optional

import typer

from src.utils.config import settings
from src.utils.logging import logger, setup_logging

app = typer.Typer(
    name="bball-predictor",
    help="International basketball game-total predictor CLI",
    add_completion=False,
)


def _init() -> None:
    setup_logging(log_level=settings.log_level, log_dir=settings.logs_dir)
    settings.ensure_dirs()


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@app.command()
def train(
    force_rebuild_gold: bool = typer.Option(False, "--rebuild-gold", help="Rebuild gold table before training"),
    seasons: Optional[list[str]] = typer.Option(None, "--season", help="Season(s) to include (repeat for multiple)"),
) -> None:
    """Run the full training pipeline (silver → gold → model → calibration)."""
    _init()
    logger.info("CLI: train")

    if force_rebuild_gold:
        typer.echo("Rebuilding gold feature table...")
        from src.pipeline.silver_to_gold import build_gold_table
        build_gold_table(seasons=seasons)

    typer.echo("Training ensemble model...")
    from src.models.trainer import train as _train
    metrics = _train()

    typer.echo("\n── Training Metrics ──────────────────────────────────")
    for k, v in metrics.items():
        typer.echo(f"  {k}: {v}")
    typer.echo("─────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# predict_today
# ---------------------------------------------------------------------------

@app.command()
def predict_today(
    leagues: Optional[list[str]] = typer.Option(None, "--league", help="League(s) to predict (repeat for multiple)"),
    output_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Fetch today's games + odds and output predictions."""
    _init()
    logger.info("CLI: predict_today")

    active_leagues = leagues or ["euroleague", "eurocup", "acb", "bsl", "bbl"]

    async def _run() -> None:
        from src.models.ensemble import BballEnsemble
        from src.scrapers.realgm import RealGMScraper
        from src.scrapers.odds import OddsClient
        from src.scrapers.injuries import InjuryScraper
        from src.injury.adjuster import InjuryAdjuster
        from src.pipeline.silver_to_gold import load_gold
        from src.api.routers.predict import _predict_games
        from datetime import datetime
        from zoneinfo import ZoneInfo

        _TZ = ZoneInfo(settings.timezone)
        now = datetime.now(_TZ)

        try:
            ensemble = BballEnsemble.load()
        except FileNotFoundError:
            typer.echo("ERROR: No trained model found. Run 'python cli.py train' first.", err=True)
            raise typer.Exit(1)

        scraper = RealGMScraper()
        odds_client = OddsClient()
        injury_scraper = InjuryScraper()
        gold_df = load_gold()
        adjuster = InjuryAdjuster.from_silver()

        all_preds = []
        for league in active_leagues:
            games = await scraper.fetch_today_schedule(league)
            if not games:
                continue
            injury_data, odds_records = await asyncio.gather(
                injury_scraper.fetch_today(league),
                odds_client.fetch_today_odds(league),
            )
            preds = _predict_games(
                ensemble=ensemble,
                games=games,
                gold_df=gold_df,
                odds_records=odds_records,
                injury_data=injury_data,
                adjuster=adjuster,
                league=league,
                now=now,
            )
            all_preds.extend(preds)

        if output_json:
            typer.echo(json.dumps([p.model_dump() for p in all_preds], indent=2, default=str))
        else:
            typer.echo(f"\n── Today's Predictions ({date.today()}) ──────────────────────")
            for p in all_preds:
                edge_str = f"  edge={p.edge:+.1f}" if p.edge is not None else ""
                line_str = f"  line={p.book_total}" if p.book_total else "  (no line)"
                typer.echo(
                    f"  [{p.league.upper():12s}] {p.match:40s}"
                    f"  model={p.model_total_mean:.1f}"
                    f"  [{p.model_total_p10:.1f}–{p.model_total_p90:.1f}]"
                    f"{line_str}{edge_str}"
                )
            typer.echo(f"─────────────────────────────────────────────────────")
            typer.echo(f"  {len(all_preds)} games predicted\n")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

@app.command()
def backtest(
    start: str = typer.Option(..., "--start", help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., "--end",   help="End date YYYY-MM-DD"),
    leagues: Optional[list[str]] = typer.Option(None, "--league"),
) -> None:
    """Backtest the trained model over a historical date range."""
    _init()
    logger.info("CLI: backtest {} → {}", start, end)

    from src.models.ensemble import BballEnsemble
    from src.pipeline.silver_to_gold import load_gold
    from src.pipeline.features import FEATURE_COLS
    from src.models.calibration import compute_coverage, report_calibration
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    import pandas as pd

    try:
        ensemble = BballEnsemble.load()
    except FileNotFoundError:
        typer.echo("ERROR: No trained model. Run 'python cli.py train' first.", err=True)
        raise typer.Exit(1)

    gold_df = load_gold()
    if gold_df.empty:
        typer.echo("ERROR: Gold table empty.", err=True)
        raise typer.Exit(1)

    gold_df["date"] = pd.to_datetime(gold_df["date"])
    start_dt = date.fromisoformat(start)
    end_dt   = date.fromisoformat(end)
    mask = (gold_df["date"].dt.date >= start_dt) & (gold_df["date"].dt.date <= end_dt)
    if leagues:
        mask &= gold_df["league"].isin(leagues)

    subset = gold_df[mask].dropna(subset=["home_points", "away_points"])
    if subset.empty:
        typer.echo("No data in specified range.", err=True)
        raise typer.Exit(1)

    X = subset[FEATURE_COLS].fillna(0).values.astype(np.float32)
    actuals = (subset["home_points"] + subset["away_points"]).values
    preds = ensemble.predict(X, game_ids=subset["game_id"].tolist())

    pred_means = np.array([p.total_mean for p in preds])
    pred_p10   = np.array([p.total_p10  for p in preds])
    pred_p90   = np.array([p.total_p90  for p in preds])

    scale = ensemble._calibration_scale
    cal_report = report_calibration(pred_means, pred_p10, pred_p90, actuals, scale)

    mae  = mean_absolute_error(actuals, pred_means)
    rmse = np.sqrt(mean_squared_error(actuals, pred_means))

    typer.echo(f"\n── Backtest Results: {start} → {end} ({len(subset)} games) ──")
    typer.echo(f"  MAE:              {mae:.2f}")
    typer.echo(f"  RMSE:             {rmse:.2f}")
    typer.echo(f"  Calibration 80%:  {cal_report['calibrated_coverage']:.1%}")
    typer.echo(f"  Interval width:   {cal_report['mean_interval_width']:.1f}")
    typer.echo(f"  Scale factor:     {scale:.4f}")

    if "book_total" in subset.columns:
        book = subset["book_total"].fillna(0).values
        valid = book > 0
        if valid.any():
            edge = pred_means[valid] - book[valid]
            hits = (edge > 0) == (actuals[valid] > book[valid])
            typer.echo(f"  Mean edge:        {np.mean(edge):+.2f}")
            typer.echo(f"  Edge hit rate:    {np.mean(hits):.1%}")
    typer.echo("─────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# scrape_test
# ---------------------------------------------------------------------------

@app.command()
def scrape_test(
    league: str = typer.Option("euroleague", "--league"),
    date_str: Optional[str] = typer.Option(None, "--date", help="YYYY-MM-DD (defaults to today)"),
) -> None:
    """Smoke-test scrapers: fetch schedule, odds, and injuries for one league."""
    _init()

    async def _run() -> None:
        from src.scrapers.realgm import RealGMScraper
        from src.scrapers.odds import OddsClient
        from src.scrapers.injuries import InjuryScraper

        scraper = RealGMScraper()
        odds_client = OddsClient()
        injury_scraper = InjuryScraper()

        typer.echo(f"Scraping {league} schedule...")
        games = await scraper.fetch_today_schedule(league)
        typer.echo(f"  → {len(games)} games found")
        for g in games[:3]:
            typer.echo(f"     {g['date']} {g['away_team']} @ {g['home_team']}")

        typer.echo(f"Fetching odds for {league}...")
        odds = await odds_client.fetch_today_odds(league)
        typer.echo(f"  → {len(odds)} odds records (source: {next(iter(odds.values())).odds_source if odds else 'n/a'})")

        typer.echo(f"Fetching injuries for {league}...")
        inj = await injury_scraper.fetch_today(league)
        typer.echo(f"  → {len(inj)} teams with unavailable players")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# build_gold
# ---------------------------------------------------------------------------

@app.command()
def build_gold(
    seasons: Optional[list[str]] = typer.Option(None, "--season"),
) -> None:
    """(Re)build the gold feature table from silver data."""
    _init()
    from src.pipeline.silver_to_gold import build_gold_table
    typer.echo("Building gold table...")
    df = build_gold_table(seasons=seasons)
    typer.echo(f"Gold table built: {len(df)} rows, {len(df.columns)} columns")


# ---------------------------------------------------------------------------
# update_actuals
# ---------------------------------------------------------------------------

@app.command()
def update_actuals() -> None:
    """Fill in post-game actuals in the daily tracker from silver box scores."""
    _init()
    import csv
    import pandas as pd
    from src.pipeline.bronze_to_silver import load_silver

    tracker_path = settings.tracking_path
    if not tracker_path.exists():
        typer.echo("No tracker file found.")
        raise typer.Exit(1)

    bs_df = load_silver("box_scores")
    if bs_df.empty:
        typer.echo("Silver box scores empty — no actuals to fill.")
        raise typer.Exit(1)

    actuals_map = bs_df.groupby("game_id")["points"].sum().to_dict()
    df = pd.read_csv(tracker_path)
    updated = 0

    for idx, row in df.iterrows():
        if pd.isna(row.get("actual_total")) or row.get("actual_total") == "":
            gid = row.get("game_id", "")
            if gid in actuals_map:
                actual = actuals_map[gid]
                df.at[idx, "actual_total"] = actual
                df.at[idx, "error"] = actual - float(row.get("model_total_mean", 0))
                updated += 1

    df.to_csv(tracker_path, index=False)
    typer.echo(f"Updated {updated} rows with actuals.")


if __name__ == "__main__":
    app()
