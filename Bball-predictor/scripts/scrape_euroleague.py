"""
Historical EuroLeague Basketball (ELB) scraper.

Covers both EuroLeague (competition code "E") and EuroCup (competition code "U")
using the official api-live.euroleague.net API (no Cloudflare, no ToS issues).
Note: RealGM is Cloudflare-protected; this API is the primary data source.

Pipeline:
  1. Fetch game list per season    → bronze/{league}/{season}/games.json
  2. Fetch box score per game      → bronze/{league}/{season}/box_{gameCode}.json
  3. Parse all into silver parquet → silver/schedules/, silver/box_scores/
  4. Build gold feature table      → gold/features.parquet

Usage:
  # EuroLeague (default)
  python scripts/scrape_euroleague.py
  python scripts/scrape_euroleague.py --season E2024

  # EuroCup
  python scripts/scrape_euroleague.py --competition U
  python scripts/scrape_euroleague.py --competition U --season U2024

  # Both together
  python scripts/scrape_euroleague.py --competition E --competition U
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import pandas as pd
import typer
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.utils.config import settings
from src.utils.logging import logger, setup_logging
from src.utils.ids import make_game_id, get_team_id

app = typer.Typer()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE = "https://api-live.euroleague.net/v2"

# Seasons per competition code
COMPETITION_META: dict[str, dict] = {
    "E": {
        "name": "euroleague",
        "seasons": {
            "E2023": "2023-24",
            "E2024": "2024-25",
            "E2025": "2025-26",
        },
    },
    "U": {
        "name": "eurocup",
        "seasons": {
            "U2023": "2023-24",
            "U2024": "2024-25",
            "U2025": "2025-26",
        },
    },
}

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "BballPredictor/1.0 (research)",
}

REQUEST_DELAY = 0.5   # seconds between API calls (polite crawling)
MAX_GAMES_PER_SEASON = 500


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=15),
    reraise=True,
)
async def _get_json(client: httpx.AsyncClient, url: str, params: dict | None = None) -> dict | list:
    resp = await client.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Step 1: Fetch game list
# ---------------------------------------------------------------------------

async def fetch_season_games(
    client: httpx.AsyncClient,
    competition: str,
    season_code: str,
    bronze_dir: Path,
    skip_existing: bool = True,
) -> list[dict]:
    """Return list of game dicts for one season (played games only)."""
    out_path = bronze_dir / f"games_{season_code}.json"

    if skip_existing and out_path.exists():
        logger.info("[{}] Game list cached — loading from {}", season_code, out_path)
        return json.loads(out_path.read_text())

    url = f"{API_BASE}/competitions/{competition}/seasons/{season_code}/games"
    params = {"gameStatus": "played", "limit": MAX_GAMES_PER_SEASON}

    logger.info("[{}] Fetching game list …", season_code)
    data = await _get_json(client, url, params)
    games = data.get("data", data) if isinstance(data, dict) else data

    # Filter to Regular Season only for clean feature building
    rs_games = [
        g for g in games
        if g.get("phaseType", {}).get("code") in ("RS", "PO", "FF")
    ]

    out_path.write_text(json.dumps(rs_games, indent=2))
    logger.info("[{}] {} games found ({} RS/PO/FF)", season_code, len(games), len(rs_games))
    return rs_games


# ---------------------------------------------------------------------------
# Step 2: Fetch box scores
# ---------------------------------------------------------------------------

async def fetch_box_score(
    client: httpx.AsyncClient,
    competition: str,
    season_code: str,
    game_code: int,
    bronze_dir: Path,
    skip_existing: bool = True,
) -> dict | None:
    """Fetch player stats for one game. Returns None on failure."""
    out_path = bronze_dir / f"box_{game_code:04d}.json"

    if skip_existing and out_path.exists():
        return json.loads(out_path.read_text())

    url = f"{API_BASE}/competitions/{competition}/seasons/{season_code}/games/{game_code}/stats"

    try:
        data = await _get_json(client, url)
        out_path.write_text(json.dumps(data, indent=2))
        return data
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.debug("[{}] game {} — no stats (404)", season_code, game_code)
        else:
            logger.warning("[{}] game {} — HTTP {}", season_code, game_code, e.response.status_code)
        return None
    except Exception as exc:
        logger.warning("[{}] game {} fetch error: {}", season_code, game_code, exc)
        return None


# ---------------------------------------------------------------------------
# Step 3: Parse bronze JSON → silver DataFrames
# ---------------------------------------------------------------------------

def _parse_minutes(time_played_secs: float) -> float:
    """Convert seconds played to decimal minutes."""
    return time_played_secs / 60.0


def parse_box_score_json(
    box_data: dict,
    game_meta: dict,
    season_code: str,
    season_label: str,
    league: str = "euroleague",
) -> list[dict]:
    """
    Parse one game's box score JSON into a list of player-level rows.

    Applies:
    - minutes: timePlayed (seconds) / 60
    - FGA = fieldGoalsAttemptedTotal
    - ORB = offensiveRebounds
    - TOV = turnovers
    - FTA = freeThrowsAttempted
    """
    rows: list[dict] = []

    # Game metadata
    game_date_str = game_meta.get("date", "")[:10]
    try:
        from datetime import date as _date
        game_date = _date.fromisoformat(game_date_str)
    except ValueError:
        return rows

    local_team_raw  = game_meta.get("local", {}).get("club", {}).get("name", "")
    road_team_raw   = game_meta.get("road", {}).get("club", {}).get("name", "")
    local_score     = game_meta.get("local", {}).get("score", 0) or 0
    road_score      = game_meta.get("road", {}).get("score", 0) or 0

    local_team_id = get_team_id(local_team_raw)
    road_team_id  = get_team_id(road_team_raw)
    game_id       = make_game_id(game_date, local_team_raw, road_team_raw)

    # Detect OT from partial scores
    partials = game_meta.get("local", {}).get("partials", {})
    extra = partials.get("extraPeriods", {})
    ot_periods = len([k for k in extra if extra[k] is not None and extra[k] != 0]) if extra else 0
    game_minutes = 40 + ot_periods * 5

    for side, team_id, team_raw, team_score in [
        ("home", local_team_id, local_team_raw, local_score),
        ("away", road_team_id,  road_team_raw,  road_score),
    ]:
        side_key = "local" if side == "home" else "road"
        players = box_data.get(side_key, {}).get("players", [])

        for player_entry in players:
            stats = player_entry.get("stats", {})
            person = player_entry.get("player", {}).get("person", {})

            minutes = _parse_minutes(float(stats.get("timePlayed", 0) or 0))
            if minutes == 0:
                continue  # did not play

            rows.append({
                "game_id":      game_id,
                "season":       season_label,
                "season_code":  season_code,
                "league":       league,
                "date":         game_date.isoformat(),
                "team_id":      team_id,
                "team_name":    team_raw,
                "team_side":    side,
                "player_name":  person.get("name", ""),
                "player_code":  person.get("code", ""),
                "minutes":      round(minutes, 3),
                "points":       float(stats.get("points", 0) or 0),
                "fga":          float(stats.get("fieldGoalsAttemptedTotal", 0) or 0),
                "fgm":          float(stats.get("fieldGoalsMadeTotal", 0) or 0),
                "fta":          float(stats.get("freeThrowsAttempted", 0) or 0),
                "ftm":          float(stats.get("freeThrowsMade", 0) or 0),
                "orb":          float(stats.get("offensiveRebounds", 0) or 0),
                "drb":          float(stats.get("defensiveRebounds", 0) or 0),
                "trb":          float(stats.get("totalRebounds", 0) or 0),
                "ast":          float(stats.get("assistances", 0) or 0),
                "tov":          float(stats.get("turnovers", 0) or 0),
                "stl":          float(stats.get("steals", 0) or 0),
                "blk":          float(stats.get("blocksFavour", 0) or 0),
                "plus_minus":   float(stats.get("plusMinus", 0) or 0),
                "valuation":    float(stats.get("valuation", 0) or 0),
                "starter":      bool(stats.get("startFive", False)),
                "team_points":  float(team_score),
                "game_minutes": float(game_minutes),
                "ot_periods":   ot_periods,
            })

    return rows


def parse_game_meta_to_schedule(game_meta: dict, season_code: str, season_label: str, league: str = "euroleague") -> dict | None:
    """Parse one game's metadata into a schedule row."""
    game_date_str = game_meta.get("date", "")[:10]
    try:
        from datetime import date as _date
        game_date = _date.fromisoformat(game_date_str)
    except ValueError:
        return None

    local_raw = game_meta.get("local", {}).get("club", {}).get("name", "")
    road_raw  = game_meta.get("road", {}).get("club", {}).get("name", "")
    local_score = game_meta.get("local", {}).get("score", 0) or 0
    road_score  = game_meta.get("road", {}).get("score", 0) or 0

    if not local_raw or not road_raw:
        return None

    return {
        "game_id":          make_game_id(game_date, local_raw, road_raw),
        "season":           season_label,
        "season_code":      season_code,
        "league":           league,
        "date":             game_date.isoformat(),
        "home_team":        local_raw,
        "away_team":        road_raw,
        "home_team_id":     get_team_id(local_raw),
        "away_team_id":     get_team_id(road_raw),
        "home_points":      float(local_score),
        "away_points":      float(road_score),
        "game_total":       float(local_score + road_score),
        "round":            game_meta.get("round", 0),
        "phase":            game_meta.get("phaseType", {}).get("code", ""),
        "played":           game_meta.get("played", False),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def scrape_season(
    competition: str,
    league_name: str,
    season_code: str,
    season_label: str,
    skip_existing: bool,
) -> tuple[list[dict], list[dict]]:
    """
    Scrape one full season. Returns (schedule_rows, player_rows).
    """
    bronze_dir = settings.bronze_dir / league_name / season_code
    bronze_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True) as client:
        # -- Game list --
        games = await fetch_season_games(client, competition, season_code, bronze_dir, skip_existing)

        schedule_rows: list[dict] = []
        player_rows: list[dict] = []
        done = 0
        errors = 0

        for game_meta in games:
            if not game_meta.get("played"):
                continue

            game_code = game_meta.get("gameCode")
            if game_code is None:
                continue

            # Schedule row (no box score needed)
            sched_row = parse_game_meta_to_schedule(game_meta, season_code, season_label, league=league_name)
            if sched_row:
                schedule_rows.append(sched_row)

            # Box score
            await asyncio.sleep(REQUEST_DELAY)
            box_data = await fetch_box_score(client, competition, season_code, game_code, bronze_dir, skip_existing)

            if box_data:
                rows = parse_box_score_json(box_data, game_meta, season_code, season_label, league=league_name)
                player_rows.extend(rows)
                done += 1
            else:
                errors += 1

            if done % 20 == 0 and done > 0:
                logger.info(
                    "[{}] Progress: {}/{} box scores fetched ({} errors)",
                    season_code, done, len(games), errors,
                )

    logger.info(
        "[{}] Done — {} schedule rows, {} player rows, {} errors",
        season_code, len(schedule_rows), len(player_rows), errors,
    )
    return schedule_rows, player_rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@app.command()
def main(
    competitions: list[str] = typer.Option(
        ["E"],
        "--competition",
        help="Competition code(s): E=EuroLeague, U=EuroCup. Repeat for multiple.",
    ),
    seasons: list[str] = typer.Option(
        [],
        "--season",
        help="Season codes (e.g. E2024, U2023). Defaults to all seasons for chosen competition(s).",
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip already-cached bronze files",
    ),
    build_gold: bool = typer.Option(
        True,
        "--build-gold/--no-build-gold",
        help="Run silver→gold pipeline after scraping",
    ),
    incremental: bool = typer.Option(
        False,
        "--incremental",
        help="Refresh current-season data only: invalidate game-list cache, keep existing box scores.",
    ),
) -> None:
    """Scrape EuroLeague / EuroCup historical data and build silver/gold tables."""
    setup_logging(log_level="INFO", log_dir=settings.logs_dir)
    settings.ensure_dirs()

    if incremental:
        # Auto-target the highest (current) season per competition if none specified
        if not seasons:
            seasons = [max(COMPETITION_META[c]["seasons"]) for c in competitions if c in COMPETITION_META]
            typer.echo(f"Incremental mode — targeting seasons: {seasons}")
        # Invalidate game-list caches so the API returns fresh results
        for comp in competitions:
            meta = COMPETITION_META.get(comp)
            if not meta:
                continue
            for sc in seasons:
                cache = settings.bronze_dir / meta["name"] / sc / f"games_{sc}.json"
                if cache.exists():
                    cache.unlink()
                    typer.echo(f"  Invalidated: {cache}")

    # Build list of (competition, league_name, season_code, season_label) tuples to scrape
    jobs: list[tuple[str, str, str, str]] = []
    for comp in competitions:
        if comp not in COMPETITION_META:
            typer.echo(f"Unknown competition: {comp!r}. Valid: {list(COMPETITION_META)}", err=True)
            continue
        meta = COMPETITION_META[comp]
        league_name = meta["name"]
        comp_seasons = meta["seasons"]

        # Filter to requested seasons (or use all if none specified)
        selected = {s: l for s, l in comp_seasons.items()
                    if not seasons or s in seasons}
        if not selected:
            typer.echo(f"No matching seasons for competition {comp}. Valid: {list(comp_seasons)}", err=True)
            continue
        for season_code, season_label in selected.items():
            jobs.append((comp, league_name, season_code, season_label))

    # Per-competition silver accumulators (keep leagues separate in silver)
    from collections import defaultdict
    all_schedules: dict[str, list[dict]] = defaultdict(list)
    all_players: dict[str, list[dict]] = defaultdict(list)

    for comp, league_name, season_code, season_label in jobs:
        typer.echo(f"\n── Scraping {league_name.upper()} {season_label} ({season_code}) ──────────────")
        sched, players = asyncio.run(
            scrape_season(comp, league_name, season_code, season_label, skip_existing)
        )
        all_schedules[league_name].extend(sched)
        all_players[league_name].extend(players)

    # -- Persist to silver (one parquet per league) --
    for league_name in set(list(all_schedules.keys()) + list(all_players.keys())):
        sched_rows = all_schedules.get(league_name, [])
        player_rows = all_players.get(league_name, [])

        if sched_rows:
            sched_df = pd.DataFrame(sched_rows).drop_duplicates("game_id")
            sched_df["date"] = pd.to_datetime(sched_df["date"])
            sched_df = sched_df.sort_values("date")
            silver_sched = settings.silver_dir / "schedules" / f"{league_name}.parquet"
            silver_sched.parent.mkdir(parents=True, exist_ok=True)
            # Merge with existing parquet if present (don't overwrite EuroLeague data)
            if silver_sched.exists():
                existing = pd.read_parquet(silver_sched)
                sched_df = pd.concat([existing, sched_df]).drop_duplicates("game_id").sort_values("date")
            sched_df.to_parquet(silver_sched, index=False)
            typer.echo(f"\n✓ Schedule saved: {len(sched_df)} games → {silver_sched}")

        if player_rows:
            player_df = pd.DataFrame(player_rows)
            player_df["date"] = pd.to_datetime(player_df["date"])

            # Oliver possessions + OT normalisation
            player_df["possessions"] = (
                player_df["fga"] - player_df["orb"] + player_df["tov"] + 0.44 * player_df["fta"]
            )
            ot_scale = (40.0 / player_df["game_minutes"]).clip(upper=1.0)
            for col in ("points", "fga", "orb", "tov", "fta", "possessions"):
                player_df[f"{col}_norm"] = player_df[col] * ot_scale
            player_df["pace_per40"] = (player_df["possessions"] / player_df["game_minutes"].clip(lower=1)) * 40
            player_df["poss_norm"] = player_df["possessions"] * ot_scale

            silver_bs = settings.silver_dir / "box_scores" / f"{league_name}.parquet"
            silver_bs.parent.mkdir(parents=True, exist_ok=True)
            if silver_bs.exists():
                existing = pd.read_parquet(silver_bs)
                player_df = pd.concat([existing, player_df]).drop_duplicates(
                    subset=["game_id", "player_code", "team_id"]
                )
            player_df.to_parquet(silver_bs, index=False)
            typer.echo(f"✓ Box scores saved: {len(player_df)} player-game rows → {silver_bs}")
            typer.echo(f"  Seasons: {sorted(player_df['season'].unique().tolist())}")
            typer.echo(f"  Unique games: {player_df['game_id'].nunique()}")

    # -- Build gold table (combines all leagues) --
    if build_gold and (any(all_schedules.values()) or any(all_players.values())):
        typer.echo("\n── Building gold feature table ──────────────────────────")
        from src.pipeline.silver_to_gold import build_gold_table
        gold_df = build_gold_table()
        if not gold_df.empty:
            typer.echo(f"✓ Gold table: {len(gold_df)} matchup rows, {len(gold_df.columns)} features")
        else:
            typer.echo("⚠  Gold table is empty — check silver data")

    typer.echo("\nDone.\n")


if __name__ == "__main__":
    app()
