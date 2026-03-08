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
    for_date: Optional[str] = typer.Option(None, "--date", help="Date to predict (YYYY-MM-DD, 'today', or 'tomorrow'). Default: today."),
    output_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Fetch scheduled games for a date and output predictions (default: today)."""
    _init()
    logger.info("CLI: predict_today")

    # Resolve target date
    from datetime import timedelta
    if for_date is None or for_date.lower() == "today":
        target_date = date.today()
    elif for_date.lower() == "tomorrow":
        target_date = date.today() + timedelta(days=1)
    else:
        try:
            target_date = date.fromisoformat(for_date)
        except ValueError:
            typer.echo(f"ERROR: Invalid date '{for_date}'. Use YYYY-MM-DD, 'today', or 'tomorrow'.", err=True)
            raise typer.Exit(1)

    active_leagues = leagues or [
        "euroleague", "eurocup", "acb", "bsl", "bbl",
        "nba", "lkl", "koris", "nbl_cz", "aba", "cba", "hung",
    ]

    async def _run() -> None:
        from src.models.ensemble import BballEnsemble
        from src.scrapers.live_schedule import LiveScheduleFetcher
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

        scraper = LiveScheduleFetcher()
        odds_client = OddsClient()
        injury_scraper = InjuryScraper()
        gold_df = load_gold()
        adjuster = InjuryAdjuster.from_silver()

        all_preds = []
        for league in active_leagues:
            games = await scraper.fetch_today(league, target_date)
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

        label = "Tomorrow's" if target_date == date.today() + timedelta(days=1) else f"{target_date}"
        if target_date == date.today():
            label = "Today's"

        if output_json:
            typer.echo(json.dumps([p.model_dump() for p in all_preds], indent=2, default=str))
        else:
            typer.echo(f"\n── {label} Predictions ({target_date}) ──────────────────────────────────────────")
            for p in all_preds:
                edge_str = f"  edge={p.edge:+.1f}" if p.edge is not None else ""
                line_str = f"  line={p.book_total:.1f}" if p.book_total else "  (no line)  "
                rec_str = f"  → {p.total_recommendation}" if p.total_recommendation else ""
                if p.away_recommendation and p.home_recommendation:
                    split_str = f"  {p.model_away_mean:.1f}({p.away_recommendation}) @ {p.model_home_mean:.1f}({p.home_recommendation})"
                else:
                    split_str = f"  {p.model_away_mean:.1f} @ {p.model_home_mean:.1f}"
                typer.echo(
                    f"  [{p.league.upper():12s}] {p.match:42s}"
                    f"  total={p.model_total_mean:.1f}"
                    f"  [{p.model_total_p10:.0f}–{p.model_total_p90:.0f}]"
                    f"{rec_str}"
                    f"{split_str}"
                    f"{line_str}{edge_str}"
                )
            typer.echo(f"──────────────────────────────────────────────────────────────────────────────────")
            typer.echo(f"  {len(all_preds)} games predicted  (split shown as: away @ home)\n")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# predictions_history  (view previously predicted games from tracker)
# ---------------------------------------------------------------------------

@app.command()
def predictions_history(
    for_date: Optional[str] = typer.Option(None, "--date", help="Date to view (YYYY-MM-DD or 'yesterday'). Default: yesterday."),
    league: Optional[str] = typer.Option(None, "--league", help="Filter by league"),
    n: int = typer.Option(50, "--n", help="Max rows to show"),
) -> None:
    """Show previously predicted games from the tracker CSV."""
    import csv
    from datetime import timedelta
    from src.utils.config import settings

    if for_date is None or for_date.lower() == "yesterday":
        target_date = date.today() - timedelta(days=1)
    elif for_date.lower() == "today":
        target_date = date.today()
    else:
        try:
            target_date = date.fromisoformat(for_date)
        except ValueError:
            typer.echo(f"ERROR: Invalid date '{for_date}'. Use YYYY-MM-DD or 'yesterday'.", err=True)
            raise typer.Exit(1)

    tracker = settings.tracking_path
    if not tracker.exists():
        typer.echo("No predictions tracker found. Run predict-today first.")
        raise typer.Exit(0)

    rows = []
    with open(tracker, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row_date = date.fromisoformat(row["date"])
            except (KeyError, ValueError):
                continue
            if row_date != target_date:
                continue
            if league and row.get("league", "").lower() != league.lower():
                continue
            rows.append(row)

    if not rows:
        typer.echo(f"No predictions found for {target_date}" + (f" ({league})" if league else "") + ".")
        raise typer.Exit(0)

    # Deduplicate by game_id (keep last prediction for each game)
    seen: dict[str, dict] = {}
    for row in rows:
        seen[row.get("game_id", row["match"])] = row
    rows = list(seen.values())[:n]

    typer.echo(f"\n── Predictions for {target_date} ({len(rows)} games) ──────────────────────────────")
    for row in rows:
        actual = row.get("actual_total", "")
        error  = row.get("error", "")
        result_str = f"  actual={actual}  err={error}" if actual else "  (no result yet)"
        book_str = f"  line={row['book_total']}" if row.get("book_total") else ""
        typer.echo(
            f"  [{row.get('league','').upper():12s}] {row.get('match',''):42s}"
            f"  total={row.get('model_total_mean',''):>6}"
            f"  [{row.get('model_total_p10',''):>3}–{row.get('model_total_p90',''):>3}]"
            f"{book_str}{result_str}"
        )
    typer.echo(f"──────────────────────────────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# predict_game  (manual matchup — no schedule scraper needed)
# ---------------------------------------------------------------------------

@app.command()
def predict_game(
    league: str = typer.Argument(..., help="League code e.g. nba, bbl, euroleague"),
    home: str = typer.Option(..., "--home", help="Home team name (as it appears in gold table)"),
    away: str = typer.Option(..., "--away", help="Away team name"),
) -> None:
    """Predict a single manually-specified matchup (bypasses live schedule fetch)."""
    _init()

    async def _run() -> None:
        from src.models.ensemble import BballEnsemble
        from src.pipeline.silver_to_gold import load_gold
        from src.injury.adjuster import InjuryAdjuster
        from src.api.routers.predict import _predict_games
        from src.utils.ids import make_game_id, get_team_id
        from datetime import datetime
        from zoneinfo import ZoneInfo

        _TZ = ZoneInfo(settings.timezone)
        now = datetime.now(_TZ)
        today = date.today()

        try:
            ensemble = BballEnsemble.load()
        except FileNotFoundError:
            typer.echo("ERROR: No trained model found. Run 'python cli.py train' first.", err=True)
            raise typer.Exit(1)

        gold_df = load_gold()
        if gold_df.empty:
            typer.echo("ERROR: Gold table empty.", err=True)
            raise typer.Exit(1)

        adjuster = InjuryAdjuster.from_silver()

        season = (
            f"{today.year}-{str(today.year + 1)[-2:]}"
            if today.month >= 10
            else f"{today.year - 1}-{str(today.year)[-2:]}"
        )
        game = {
            "game_id":      make_game_id(today, home, away),
            "date":         today.isoformat(),
            "home_team":    home,
            "away_team":    away,
            "home_team_id": get_team_id(home),
            "away_team_id": get_team_id(away),
            "league":       league,
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
            league=league,
            now=now,
        )

        if not preds:
            typer.echo(
                f"ERROR: No historical data found for '{home}' or '{away}' in {league}.\n"
                f"Tip: check exact names with: python cli.py list-teams --league {league}",
                err=True,
            )
            raise typer.Exit(1)

        p = preds[0]
        total_rec_str = f"  → {p.total_recommendation}" if p.total_recommendation else ""
        if p.away_recommendation and p.home_recommendation:
            split_str = f"{p.model_away_mean:.1f}({p.away_recommendation}) @ {p.model_home_mean:.1f}({p.home_recommendation})"
        else:
            split_str = f"{p.model_away_mean:.1f} @ {p.model_home_mean:.1f}"
        typer.echo(f"\n── Manual Prediction ─────────────────────────────────────────────")
        typer.echo(f"  [{league.upper():12s}]  {p.match}")
        typer.echo(f"  Total:      {p.model_total_mean:.1f}  [{p.model_total_p10:.0f}–{p.model_total_p90:.0f}]{total_rec_str}")
        if p.book_total:
            typer.echo(f"  Line:       {p.book_total:.1f}  (edge={p.edge:+.1f})")
        typer.echo(f"  Split:      {split_str}  (away @ home)")
        typer.echo(f"  Confidence: {p.confidence:.1%}")
        if p.home_pts_l5:
            typer.echo(f"  Home L5:    scores {p.home_pts_l5:.1f},  allows {p.home_pts_allowed_l5:.1f}")
        if p.away_pts_l5:
            typer.echo(f"  Away L5:    scores {p.away_pts_l5:.1f},  allows {p.away_pts_allowed_l5:.1f}")
        typer.echo(f"─────────────────────────────────────────────────────────────────\n")

    asyncio.run(_run())


@app.command()
def list_teams(
    league: str = typer.Option(..., "--league", help="League code"),
) -> None:
    """List all team names available in the gold table for a given league."""
    _init()
    from src.pipeline.silver_to_gold import load_gold

    gold_df = load_gold()
    if gold_df.empty:
        typer.echo("Gold table is empty.")
        raise typer.Exit(1)

    subset = gold_df[gold_df["league"] == league] if "league" in gold_df.columns else gold_df
    if subset.empty:
        typer.echo(f"No data found for league '{league}'.")
        raise typer.Exit(1)

    # Gold table stores slugified IDs (e.g. "boston-celtics"). Convert back to
    # title-cased display names so users know what to type in the predict-game form.
    homes = subset["home_team_id"].dropna().unique() if "home_team_id" in subset.columns else []
    aways = subset["away_team_id"].dropna().unique() if "away_team_id" in subset.columns else []
    team_ids = sorted(set(list(homes)) | set(list(aways)))
    teams = [tid.replace("-", " ").title() for tid in team_ids]
    typer.echo(f"\nTeams in {league.upper()} ({len(teams)} teams)  — use these names exactly:")
    for t in teams:
        typer.echo(f"  {t}")
    typer.echo()


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

    X = subset.reindex(columns=FEATURE_COLS, fill_value=0).values.astype(np.float32)
    actuals = (subset["home_points"] + subset["away_points"]).values
    leagues_list = subset["league"].tolist() if "league" in subset.columns else None
    preds = ensemble.predict(X, game_ids=subset["game_id"].tolist(), leagues=leagues_list)

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
        from src.scrapers.live_schedule import LiveScheduleFetcher
        from src.scrapers.odds import OddsClient
        from src.scrapers.injuries import InjuryScraper

        scraper = LiveScheduleFetcher()
        odds_client = OddsClient()
        injury_scraper = InjuryScraper()

        typer.echo(f"Scraping {league} schedule...")
        games = await scraper.fetch_today(league)
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
    from pathlib import Path
    typer.echo("Building gold table...")
    df = build_gold_table(seasons=seasons)
    typer.echo(f"Gold table built: {len(df)} rows, {len(df.columns)} columns")
    # Apply OddsPortal fuzzy odds matching if silver odds exist
    silver_dir = Path(settings.data_dir) / "silver"
    gold_path = Path(settings.data_dir) / "gold" / "features.parquet"
    if (silver_dir / "odds.parquet").exists() and gold_path.exists():
        from scripts.scrape_odds_oddsportal import patch_gold_with_alt_ids
        patch_gold_with_alt_ids(silver_dir, gold_path)


# ---------------------------------------------------------------------------
# refresh_data
# ---------------------------------------------------------------------------

@app.command()
def refresh_data(
    retrain: bool = typer.Option(False, "--retrain", help="Retrain model after refreshing data"),
) -> None:
    """Refresh silver data with the latest games from the current season, rebuild gold."""
    import subprocess
    import sys
    from pathlib import Path as _Path

    _init()
    scripts_dir = _Path(__file__).parent / "scripts"

    typer.echo("── EuroLeague + EuroCup ─────────────────────────────────")
    subprocess.run(
        [sys.executable, str(scripts_dir / "scrape_euroleague.py"),
         "--competition", "E", "--competition", "U",
         "--incremental", "--no-build-gold"],
        check=True,
    )

    typer.echo("\n── ACB + BBL + BSL + NBA + LKL + KORIS + NBL_CZ + ABA + CBA + HUNG ──────────")
    subprocess.run(
        [sys.executable, str(scripts_dir / "scrape_national_leagues.py"),
         "--league", "acb", "--league", "bbl", "--league", "bsl",
         "--league", "nba", "--league", "lkl", "--league", "koris",
         "--league", "nbl_cz", "--league", "aba", "--league", "cba",
         "--league", "hung",
         "--incremental", "--no-build-gold"],
        check=True,
    )

    typer.echo("\n── Rebuilding gold feature table ────────────────────────")
    from src.pipeline.silver_to_gold import build_gold_table
    df = build_gold_table()
    typer.echo(f"Gold table: {len(df)} rows, {len(df.columns)} columns")

    if retrain:
        typer.echo("\n── Retraining ensemble model ────────────────────────────")
        from src.models.trainer import train as _train
        metrics = _train()
        for k, v in metrics.items():
            typer.echo(f"  {k}: {v}")

    typer.echo("\nDone.\n")


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


# ---------------------------------------------------------------------------
# update_biases
# ---------------------------------------------------------------------------

@app.command()
def update_biases(
    min_games: int = typer.Option(5, "--min-games", help="Minimum completed games per league before adjusting bias"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print adjustments without saving"),
) -> None:
    """
    Update per-league bias corrections from live tracker results.

    Reads completed games from the daily tracker, computes residual error
    per league (actual - model, after existing bias correction), and patches
    the running ensemble so future predictions are corrected without a full retrain.

    Run after: python cli.py update-actuals
    """
    _init()
    import json
    from datetime import datetime
    from pathlib import Path
    import pandas as pd
    from src.models.ensemble import BballEnsemble

    tracker_path = settings.tracking_path
    if not tracker_path.exists():
        typer.echo("No tracker file found — run predict-today first.", err=True)
        raise typer.Exit(1)

    df = pd.read_csv(tracker_path)
    filled = df[df["actual_total"].notna()].copy()

    if filled.empty:
        typer.echo("No completed games in tracker — run update-actuals first.")
        raise typer.Exit(1)

    # error = actual - model (positive = under-predicted, negative = over-predicted)
    league_stats = (
        filled.groupby("league")["error"]
        .agg(mean_error="mean", games="count")
        .reset_index()
    )

    typer.echo(f"\nTracker residuals ({len(filled)} completed games):")
    typer.echo(f"{'League':<14} {'Games':>5}  {'Mean error':>10}  {'Direction':>12}")
    typer.echo("─" * 50)
    for _, row in league_stats.iterrows():
        direction = "under-predicting" if row["mean_error"] > 0 else "over-predicting"
        typer.echo(
            f"  {row['league']:<12} {int(row['games']):>5}  {row['mean_error']:>+10.2f}  {direction}"
        )

    # Leagues with enough games to update
    to_update = league_stats[league_stats["games"] >= min_games]
    skipped = league_stats[league_stats["games"] < min_games]["league"].tolist()
    if skipped:
        typer.echo(f"\nSkipped (< {min_games} games): {', '.join(skipped)}")

    if to_update.empty:
        typer.echo(f"\nNo leagues have ≥ {min_games} completed games yet. Nothing updated.")
        raise typer.Exit(0)

    # Load ensemble and patch biases
    ensemble = BballEnsemble.load()
    log_entries = []

    typer.echo(f"\nBias adjustments:")
    typer.echo(f"{'League':<14} {'Old total bias':>14}  {'Adjustment':>10}  {'New total bias':>14}")
    typer.echo("─" * 60)

    for _, row in to_update.iterrows():
        league = row["league"]
        residual = float(row["mean_error"])

        # Split residual equally between home and away sides
        half = residual / 2

        old_home = ensemble._league_home_bias.get(league, 0.0)
        old_away = ensemble._league_away_bias.get(league, 0.0)
        old_total = old_home + old_away

        new_home = old_home + half
        new_away = old_away + half
        new_total = new_home + new_away

        typer.echo(
            f"  {league:<12} {old_total:>+14.3f}  {residual:>+10.3f}  {new_total:>+14.3f}"
        )

        if not dry_run:
            ensemble._league_home_bias[league] = round(new_home, 4)
            ensemble._league_away_bias[league] = round(new_away, 4)

        log_entries.append({
            "league": league,
            "games_used": int(row["games"]),
            "residual": round(residual, 4),
            "old_home_bias": round(old_home, 4),
            "old_away_bias": round(old_away, 4),
            "new_home_bias": round(new_home, 4),
            "new_away_bias": round(new_away, 4),
        })

    if dry_run:
        typer.echo("\n[dry-run] No changes saved.")
        return

    ensemble.save()

    # Persist log
    log_path = Path(settings.models_dir) / "bias_update_log.json"
    existing_log = json.loads(log_path.read_text()) if log_path.exists() else []
    existing_log.append({
        "timestamp": datetime.now().isoformat(),
        "completed_games_in_tracker": len(filled),
        "updates": log_entries,
    })
    log_path.write_text(json.dumps(existing_log, indent=2))

    typer.echo(f"\nEnsemble updated and saved. Log → {log_path}")
    typer.echo("Next predictions will use the adjusted biases automatically.")


if __name__ == "__main__":
    app()
