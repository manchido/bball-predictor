"""
Historical national league scraper: ACB (Spain), BBL (Germany), BSL (Turkey),
NBA, LKL (Lithuania), Korisliiga (Finland), NBL Czech Republic,
ABA League (Balkans), CBA (China), Hungary NB I/A.

Data sources
────────────
  ACB  → www.acb.com  (official website, HTML)
         Calendar page:  /calendario/index/temporada_id/{year}
         Box score page: /partido/estadisticas/id/{game_id}

  All SofaScore leagues → SofaScore unofficial API  (api.sofascore.com)

         SofaScore tournament / season IDs (verified 2025-03-03 / 2026-03-05):
           BBL  tid=227   23/24=52951  24/25=65031  25/26=79994
           BSL  tid=519   23/24=54528  24/25=65808  25/26=81036
           NBA  tid=132   23/24=54105  24/25=65360  25/26=80229
           LKL  tid=975   23/24=54106  24/25=65649  25/26=80356
           KORIS tid=226  23/24=54554  24/25=63705  25/26=79938
           NBLCZ tid=250  23/24=53056  24/25=63530  25/26=77912
           ABA  tid=235   23/24=53473  24/25=61743  25/26=80150
           CBA  tid=1566  23/24=55486  24/25=67166  25/26=85375
           HUNG tid=10594 23/24=55019  24/25=66422  25/26=79941

Pipeline (identical silver format to scrape_euroleague.py):
  1. Fetch season schedule     → bronze/{league}/{season}/schedule.json
  2. Fetch game box scores     → bronze/{league}/{season}/box_{game_code}.json
  3. Parse → silver parquet    → silver/schedules/{league}.parquet
                               → silver/box_scores/{league}.parquet
  4. Build gold feature table  → gold/features.parquet

Usage
─────
  # Scrape all leagues and seasons:
  python scripts/scrape_national_leagues.py

  # Single league:
  python scripts/scrape_national_leagues.py --league acb

  # Single season:
  python scripts/scrape_national_leagues.py --league bbl --season BBL2024

  # Don't rebuild gold table:
  python scripts/scrape_national_leagues.py --no-build-gold

ACB season IDs
──────────────
  temporada_id = start year of season
    2023 → 2023-24 Liga Endesa
    2024 → 2024-25 Liga Endesa
    2025 → 2025-26 Liga Endesa
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import pandas as pd
import typer
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.utils.config import settings
from src.utils.ids import make_game_id, get_team_id
from src.utils.logging import logger, setup_logging

app = typer.Typer()

# ---------------------------------------------------------------------------
# League configuration
# ---------------------------------------------------------------------------

# ACB season IDs (temporada_id = start year of season)
ACB_SEASON_IDS: dict[str, tuple[str, int]] = {
    "ACB2023": ("2023-24", 2023),
    "ACB2024": ("2024-25", 2024),
    "ACB2025": ("2025-26", 2025),
}

# SofaScore configuration for BBL, BSL and expanded leagues
# IDs verified 2025-03-03 via api.sofascore.com/api/v1/category/{cat_id}/unique-tournaments
# New league IDs verified 2026-03-05 via probe_seasons.py
SOFASCORE_LEAGUES: dict[str, dict] = {
    "bbl": {
        "tournament_id": 227,   # Germany BBL
        "league_name": "bbl",
        "seasons": {
            "BBL2023": ("2023-24", 52951),
            "BBL2024": ("2024-25", 65031),
            "BBL2025": ("2025-26", 79994),
        },
    },
    "bsl": {
        "tournament_id": 519,   # Turkish Basketball Super League
        "league_name": "bsl",
        "seasons": {
            "BSL2023": ("2023-24", 54528),
            "BSL2024": ("2024-25", 65808),
            "BSL2025": ("2025-26", 81036),
        },
    },
    "nba": {
        "tournament_id": 132,   # NBA
        "league_name": "nba",
        "seasons": {
            "NBA2023": ("2023-24", 54105),
            "NBA2024": ("2024-25", 65360),
            "NBA2025": ("2025-26", 80229),
        },
    },
    "lkl": {
        "tournament_id": 975,   # Lithuanian Basketball League (LKL)
        "league_name": "lkl",
        "seasons": {
            "LKL2023": ("2023-24", 54106),
            "LKL2024": ("2024-25", 65649),
            "LKL2025": ("2025-26", 80356),
        },
    },
    "koris": {
        "tournament_id": 226,   # Korisliiga (Finland)
        "league_name": "koris",
        "seasons": {
            "KORIS2023": ("2023-24", 54554),
            "KORIS2024": ("2024-25", 63705),
            "KORIS2025": ("2025-26", 79938),
        },
    },
    "nbl_cz": {
        "tournament_id": 250,   # NBL Czech Republic
        "league_name": "nbl_cz",
        "seasons": {
            "NBLCZ2023": ("2023-24", 53056),
            "NBLCZ2024": ("2024-25", 63530),
            "NBLCZ2025": ("2025-26", 77912),
        },
    },
    "aba": {
        "tournament_id": 235,   # ABA League (Adriatic Basketball Association)
        "league_name": "aba",
        "seasons": {
            "ABA2023": ("2023-24", 53473),
            "ABA2024": ("2024-25", 61743),
            "ABA2025": ("2025-26", 80150),
        },
    },
    "cba": {
        "tournament_id": 1566,  # CBA (Chinese Basketball Association)
        "league_name": "cba",
        "seasons": {
            "CBA2023": ("2023-24", 55486),
            "CBA2024": ("2024-25", 67166),
            "CBA2025": ("2025-26", 85375),
        },
    },
    "hung": {
        "tournament_id": 10594, # Hungarian NB I/A
        "league_name": "hung",
        "seasons": {
            "HUNG2023": ("2023-24", 55019),
            "HUNG2024": ("2024-25", 66422),
            "HUNG2025": ("2025-26", 79941),
        },
    },
}

ACB_BASE       = "https://www.acb.com"
SOFASCORE_BASE = "https://api.sofascore.com/api/v1"

HEADERS_BROWSER = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
# SofaScore: browser-style headers + Referer required to avoid 403
HEADERS_SOFASCORE = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://www.sofascore.com/",
}

ACB_DELAY        = 1.5    # seconds between ACB page requests (polite)
SOFASCORE_DELAY  = 1.2    # seconds between SofaScore API calls


# ---------------------------------------------------------------------------
# Shared HTTP helpers
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=3, max=20),
    reraise=True,
)
async def _get_html(
    client: httpx.AsyncClient, url: str, params: dict | None = None
) -> str:
    resp = await client.get(url, params=params, timeout=25)
    resp.raise_for_status()
    return resp.text


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    reraise=True,
)
async def _get_json(
    client: httpx.AsyncClient, url: str, params: dict | None = None
) -> dict | list:
    resp = await client.get(url, params=params, timeout=25)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# ACB scraper
# ---------------------------------------------------------------------------

async def acb_scrape_season(
    client: httpx.AsyncClient,
    season_code: str,
    season_label: str,
    temporada_id: int,
    bronze_dir: Path,
    skip_existing: bool,
) -> tuple[list[dict], list[dict]]:
    """
    Scrape one ACB Liga Endesa season.

    1. Fetch calendar page → extract all played game IDs
    2. For each game, fetch box score page → parse player stats + metadata
    """
    schedule_rows: list[dict] = []
    player_rows: list[dict] = []

    # ── Step 1: get game IDs from calendar ──────────────────────────────────
    cal_cache = bronze_dir / f"calendar_{season_code}.json"
    if skip_existing and cal_cache.exists():
        game_ids = json.loads(cal_cache.read_text())
        logger.info("[ACB {}] Loaded {} game IDs from cache", season_code, len(game_ids))
    else:
        cal_url = f"{ACB_BASE}/calendario/index/temporada_id/{temporada_id}"
        logger.info("[ACB {}] Fetching calendar: {}", season_code, cal_url)
        try:
            html = await _get_html(client, cal_url)
            # Find all game IDs (links of form /partido/estadisticas/id/{id})
            game_ids = sorted(set(re.findall(
                r"/partido/estadisticas/id/(\d{4,8})", html
            )))
            cal_cache.write_text(json.dumps(game_ids))
            logger.info("[ACB {}] Found {} game IDs", season_code, len(game_ids))
        except Exception as exc:
            logger.error("[ACB {}] Calendar fetch failed: {}", season_code, exc)
            return [], []

    if not game_ids:
        logger.warning("[ACB {}] No game IDs found", season_code)
        return [], []

    # ── Step 2: box scores ───────────────────────────────────────────────────
    done = errors = 0
    for game_id_raw in game_ids:
        box_cache = bronze_dir / f"box_{game_id_raw}.json"

        if skip_existing and box_cache.exists():
            box_meta = json.loads(box_cache.read_text())
        else:
            await asyncio.sleep(ACB_DELAY)
            url = f"{ACB_BASE}/partido/estadisticas/id/{game_id_raw}"
            try:
                html = await _get_html(client, url)
                box_meta = _parse_acb_box_html(
                    html, game_id_raw, season_code, season_label
                )
                box_cache.write_text(json.dumps(box_meta, indent=2))
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.debug("[ACB {}] game {} — 404", season_code, game_id_raw)
                else:
                    logger.warning("[ACB {}] game {} HTTP {}", season_code,
                                   game_id_raw, e.response.status_code)
                errors += 1
                continue
            except Exception as exc:
                logger.warning("[ACB {}] game {} error: {}", season_code, game_id_raw, exc)
                errors += 1
                continue

        if box_meta and box_meta.get("player_rows"):
            schedule_rows.append(box_meta["sched_row"])
            player_rows.extend(box_meta["player_rows"])
            done += 1
        else:
            errors += 1

        if done % 30 == 0 and done > 0:
            logger.info("[ACB {}] Progress: {}/{} games done, {} errors",
                        season_code, done, len(game_ids), errors)

    logger.info(
        "[ACB {}] Complete: {} schedule rows, {} player rows, {} errors",
        season_code, len(schedule_rows), len(player_rows), errors,
    )
    return schedule_rows, player_rows


def _parse_acb_box_html(
    html: str,
    game_id_raw: str,
    season_code: str,
    season_label: str,
) -> dict:
    """
    Parse one ACB game box score HTML page into a dict with:
      sched_row:   one schedule record
      player_rows: list of player stat dicts

    ACB page structure (verified 2025-03-03):
    ──────────────────────────────────────────
    div.datos_fecha : 'Jornada 4 | 08/10/2023 | 12:30 | ...'
    div.cabecera_partido :
      children include home_team_name + 'Jornada N - Liga Endesa' + away_team_name
    table[0] : quarter scores  (row0=home abbr+quarters, row1=away abbr+quarters)
    table[1] : home player stats
    table[2] : away player stats

    Player stat column indices:
      0  jersey#   1  name   2  min(mm:ss)   3  pts
      4  2ptM/A    5  2pt%   6  3ptM/A       7  3pt%
      8  FTM/FTA   9  FT%   10  REB         11  ORB+DRB
     12  AST      13  STL   14  TOV         ...
     21  +/-      22  VAL
    """
    soup = BeautifulSoup(html, "lxml")
    result: dict = {"sched_row": None, "player_rows": []}

    # ── Date ──────────────────────────────────────────────────────────────────
    date_div = soup.select_one("div.datos_fecha")
    game_date = None
    if date_div:
        m = re.search(r"(\d{2}/\d{2}/\d{4})", date_div.get_text())
        if m:
            try:
                from datetime import datetime as _dt
                game_date = _dt.strptime(m.group(1), "%d/%m/%Y").date()
            except ValueError:
                pass

    if not game_date:
        logger.debug("[ACB] game {} — no date found, skipping", game_id_raw)
        return result

    # ── Team names + scores ───────────────────────────────────────────────────
    # Primary: div.cabecera_partido contains "{home}Jornada N - Liga Endesa{away}"
    cab = soup.select_one("div.cabecera_partido")
    home_name = away_name = ""
    if cab:
        text = cab.get_text(strip=True)
        # Split at "Jornada" keyword
        m_jorn = re.search(r"^(.+?)Jornada\s+\d+\s*-\s*[^\d]+?([A-ZÀ-ÿ].+?)$", text)
        if m_jorn:
            home_name = m_jorn.group(1).strip()
            away_name = m_jorn.group(2).strip()

    # Fallback: quarter score table abbreviations + span.nombre_largo from
    # article.partido elements in the same page
    tables = soup.select("table")
    home_score = away_score = 0.0
    ot_periods = 0

    if tables:
        t0 = tables[0]
        rows = t0.select("tr")
        if len(rows) >= 3:
            home_cells = [td.get_text(strip=True) for td in rows[1].select("td")]
            away_cells = [td.get_text(strip=True) for td in rows[2].select("td")]

            # Quarter scores start at index 2
            def _sum_quarters(cells: list[str]) -> tuple[float, int]:
                scores = []
                for c in cells[2:]:
                    c = c.strip()
                    if c.isdigit():
                        scores.append(int(c))
                extra = max(0, len(scores) - 4)
                return float(sum(scores)), extra

            home_score, ot_periods = _sum_quarters(home_cells)
            away_score, _         = _sum_quarters(away_cells)

    game_minutes = 40 + ot_periods * 5

    # If team names still missing, try to get from nearby elements
    if not home_name:
        # Try to reconstruct from page title
        title = soup.select_one("title")
        if title:
            t = title.get_text(strip=True)
            m = re.search(r"([A-Z][\w\s]+)\s+\d{2,3}\s*[-–]\s*\d{2,3}\s+([A-Z][\w\s]+)", t)
            if m:
                home_name, away_name = m.group(1).strip(), m.group(2).strip()

    if not home_name or not away_name:
        logger.debug("[ACB] game {} — could not parse team names, skipping", game_id_raw)
        return result

    game_id = make_game_id(game_date, home_name, away_name)

    # ── Player stats ──────────────────────────────────────────────────────────
    def _parse_frac(s: str) -> tuple[float, float]:
        """Parse 'M/A' string. Returns (made, attempted)."""
        parts = s.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0] or 0), float(parts[1] or 0)
            except ValueError:
                pass
        return 0.0, 0.0

    def _parse_split(s: str, sep: str = "+") -> tuple[float, float]:
        """Parse 'X+Y' or 'X-Y' string."""
        parts = re.split(r"[+\-]", str(s))
        if len(parts) == 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                pass
        return 0.0, 0.0

    def _parse_min(s: str) -> float:
        """Parse mm:ss to decimal minutes."""
        if ":" in str(s):
            p = str(s).split(":")
            try:
                return float(p[0]) + float(p[1]) / 60
            except ValueError:
                pass
        try:
            return float(s)
        except (ValueError, TypeError):
            return 0.0

    player_rows: list[dict] = []

    # Tables[1] = home team players, tables[2] = away team players
    for side_idx, side in enumerate(["home", "away"]):
        tbl_idx = 1 + side_idx
        if tbl_idx >= len(tables):
            continue
        tbl = tables[tbl_idx]
        team_name = home_name if side == "home" else away_name
        team_id   = get_team_id(team_name)
        team_score = home_score if side == "home" else away_score

        for row in tbl.select("tbody tr"):
            cells = [td.get_text(strip=True) for td in row.select("td")]
            if len(cells) < 15:
                continue
            player_name = cells[1] if len(cells) > 1 else ""
            if not player_name or player_name.lower() in ("equipo", "total", ""):
                continue

            minutes = _parse_min(cells[2])
            if minutes < 0.1:
                continue

            pts   = float(cells[3] or 0) if cells[3].lstrip("-").isdigit() else 0.0
            _, fg2a = _parse_frac(cells[4]) if len(cells) > 4 else (0.0, 0.0)
            _, fg3a = _parse_frac(cells[6]) if len(cells) > 6 else (0.0, 0.0)
            ftm, fta = _parse_frac(cells[8]) if len(cells) > 8 else (0.0, 0.0)
            orb, drb = _parse_split(cells[11]) if len(cells) > 11 else (0.0, 0.0)

            # Total rebounds from cell 10 (more reliable than sum of 11)
            try:
                trb = float(cells[10] or 0)
            except ValueError:
                trb = orb + drb

            try:
                ast = float(cells[12] or 0)
            except ValueError:
                ast = 0.0
            try:
                stl = float(cells[13] or 0)
            except ValueError:
                stl = 0.0
            try:
                tov = float(cells[14] or 0)
            except ValueError:
                tov = 0.0

            pm = val = 0.0
            if len(cells) > 21:
                try:
                    pm = float(cells[21])
                except ValueError:
                    pass
            if len(cells) > 22:
                try:
                    val = float(cells[22])
                except ValueError:
                    pass

            fga_total = fg2a + fg3a

            player_rows.append({
                "game_id":      game_id,
                "season":       season_label,
                "season_code":  season_code,
                "league":       "acb",
                "date":         game_date.isoformat(),
                "team_id":      team_id,
                "team_name":    team_name,
                "team_side":    side,
                "player_name":  player_name,
                "player_code":  cells[0],
                "minutes":      round(minutes, 3),
                "points":       pts,
                "fga":          fga_total,
                "fgm":          0.0,  # not separately available
                "fta":          fta,
                "ftm":          ftm,
                "orb":          orb,
                "drb":          drb,
                "trb":          trb,
                "ast":          ast,
                "tov":          tov,
                "stl":          stl,
                "blk":          0.0,
                "plus_minus":   pm,
                "valuation":    val,
                "starter":      False,
                "team_points":  team_score,
                "game_minutes": float(game_minutes),
                "ot_periods":   ot_periods,
            })

    if not player_rows:
        return result

    sched_row = {
        "game_id":      game_id,
        "season":       season_label,
        "season_code":  season_code,
        "league":       "acb",
        "date":         game_date.isoformat(),
        "home_team":    home_name,
        "away_team":    away_name,
        "home_team_id": get_team_id(home_name),
        "away_team_id": get_team_id(away_name),
        "home_points":  home_score,
        "away_points":  away_score,
        "game_total":   home_score + away_score,
        "phase":        "RS",
        "round":        0,
        "played":       True,
    }

    result["sched_row"]   = sched_row
    result["player_rows"] = player_rows
    return result


# ---------------------------------------------------------------------------
# SofaScore scraper  (BBL + BSL)
# ---------------------------------------------------------------------------

async def sofascore_scrape_season(
    client: httpx.AsyncClient,
    league: str,
    tournament_id: int,
    season_code: str,
    season_label: str,
    season_id: int,
    bronze_dir: Path,
    skip_existing: bool,
) -> tuple[list[dict], list[dict]]:
    """
    Scrape one BBL / BSL season via SofaScore's unofficial API.

    Endpoint summary:
      Events (paginated): GET /unique-tournament/{t_id}/season/{s_id}/events/last/{page}
        30 events/page; hasNextPage=True while more remain
      Event detail:       GET /event/{event_id}
        homeScore / awayScore dicts with period1..period4 [+period5..] for OT
      Lineups + stats:    GET /event/{event_id}/lineups
        home.players[].statistics — secondsPlayed, twoPointAttempts,
        threePointAttempts, freeThrowAttempts, freeThrowsMade,
        offensiveRebounds, defensiveRebounds, turnovers, assists, steals, blocks
    """
    from datetime import datetime as _dt, timezone as _tz

    schedule_rows: list[dict] = []
    player_rows:   list[dict] = []

    # ── Step 1: fetch all event IDs for the season ───────────────────────────
    events_cache = bronze_dir / f"schedule_{season_code}.json"
    if skip_existing and events_cache.exists():
        all_events = json.loads(events_cache.read_text())
        logger.info("[{} {}] Loaded {} events from cache", league.upper(), season_code, len(all_events))
    else:
        all_events = await _sofascore_fetch_events(
            client, tournament_id, season_id, league, season_code
        )
        if all_events:
            events_cache.write_text(json.dumps(all_events, indent=2))

    finished_events = [
        e for e in all_events
        if e.get("status", {}).get("type") == "finished"
    ]
    logger.info(
        "[{} {}] {} total events, {} finished",
        league.upper(), season_code, len(all_events), len(finished_events),
    )

    if not finished_events:
        logger.warning("[{} {}] No finished games — skipping", league.upper(), season_code)
        return [], []

    # ── Step 2: box scores ───────────────────────────────────────────────────
    done = errors = 0
    for ev_meta in finished_events:
        event_id = str(ev_meta["id"])
        box_path = bronze_dir / f"box_{event_id}.json"

        if skip_existing and box_path.exists():
            box_data = json.loads(box_path.read_text())
        else:
            await asyncio.sleep(SOFASCORE_DELAY)
            try:
                lineup_url = f"{SOFASCORE_BASE}/event/{event_id}/lineups"
                lineups = await _get_json(client, lineup_url)
                box_data = {"event_meta": ev_meta, "lineups": lineups}
                box_path.write_text(json.dumps(box_data, indent=2))
            except httpx.HTTPStatusError as e:
                logger.debug("[{} {}] event {} HTTP {}", league, season_code,
                             event_id, e.response.status_code)
                errors += 1
                continue
            except Exception as exc:
                logger.warning("[{} {}] event {} error: {}", league, season_code, event_id, exc)
                errors += 1
                continue

        rows, sched_row = _parse_sofascore_box(
            box_data, season_code, season_label, league
        )
        player_rows.extend(rows)
        if sched_row:
            schedule_rows.append(sched_row)
            done += 1
        else:
            errors += 1

        if done % 30 == 0 and done > 0:
            logger.info("[{} {}] Progress: {}/{} games done, {} errors",
                        league.upper(), season_code, done, len(finished_events), errors)

    logger.info(
        "[{} {}] Complete: {} schedule rows, {} player rows, {} errors",
        league.upper(), season_code, len(schedule_rows), len(player_rows), errors,
    )
    return schedule_rows, player_rows


async def _sofascore_fetch_events(
    client: httpx.AsyncClient,
    tournament_id: int,
    season_id: int,
    league: str,
    season_code: str,
) -> list[dict]:
    """Paginate through all events for a SofaScore season."""
    all_events: list[dict] = []
    page = 0
    while True:
        url = f"{SOFASCORE_BASE}/unique-tournament/{tournament_id}/season/{season_id}/events/last/{page}"
        try:
            data = await _get_json(client, url)
        except Exception as exc:
            logger.warning("[{} {}] Events page {} error: {}", league.upper(), season_code, page, exc)
            break

        events = data.get("events", [])
        all_events.extend(events)
        has_next = data.get("hasNextPage", False)
        logger.debug("[{} {}] Events page {}: {} events", league.upper(), season_code, page, len(events))

        if not has_next:
            break
        page += 1
        await asyncio.sleep(SOFASCORE_DELAY)

    return all_events


def _parse_sofascore_box(
    box_data: dict,
    season_code: str,
    season_label: str,
    league: str,
) -> tuple[list[dict], dict | None]:
    """
    Parse SofaScore lineup JSON into player rows + schedule row.

    box_data structure:
      event_meta: the event dict from the events/last endpoint
      lineups:    GET /event/{id}/lineups response
        confirmed, statisticalVersion
        home.players[].statistics  (secondsPlayed, twoPointAttempts, ...)
        away.players[].statistics
    """
    from datetime import datetime as _dt, timezone as _tz

    event_meta = box_data.get("event_meta", {})
    lineups    = box_data.get("lineups", {})

    # ── Date from startTimestamp ─────────────────────────────────────────────
    ts = event_meta.get("startTimestamp")
    if not ts:
        return [], None
    try:
        game_date = _dt.fromtimestamp(ts, tz=_tz.utc).date()
    except Exception:
        return [], None

    # ── Team names + scores ─────────────────────────────────────────────────
    home_name  = event_meta.get("homeTeam", {}).get("name", "")
    away_name  = event_meta.get("awayTeam", {}).get("name", "")
    if not home_name or not away_name:
        return [], None

    home_score_dict = event_meta.get("homeScore", {})
    away_score_dict = event_meta.get("awayScore", {})
    home_score = float(home_score_dict.get("current", 0) or 0)
    away_score = float(away_score_dict.get("current", 0) or 0)

    # OT detection: extra periods beyond period4 in homeScore dict
    period_keys = [k for k in home_score_dict if k.startswith("period") and k[6:].isdigit()]
    ot_periods  = max(0, len(period_keys) - 4)
    game_minutes = 40 + ot_periods * 5

    game_id = make_game_id(game_date, home_name, away_name)

    # ── Player stats from lineups ────────────────────────────────────────────
    player_rows: list[dict] = []

    for side in ("home", "away"):
        side_data  = lineups.get(side, {})
        team_name  = home_name if side == "home" else away_name
        team_id    = get_team_id(team_name)
        team_score = home_score if side == "home" else away_score
        players    = side_data.get("players", [])

        for p in players:
            stats = p.get("statistics")
            if not stats:
                continue

            secs = float(stats.get("secondsPlayed", 0) or 0)
            minutes = secs / 60.0
            if minutes < 0.1:
                continue

            fg2a = float(stats.get("twoPointAttempts", 0) or 0)
            fg3a = float(stats.get("threePointAttempts", 0) or 0)
            fga  = fg2a + fg3a
            fgm  = float(stats.get("fieldGoalsMade", 0) or 0)
            fta  = float(stats.get("freeThrowAttempts", 0) or 0)
            ftm  = float(stats.get("freeThrowsMade", 0) or 0)
            orb  = float(stats.get("offensiveRebounds", 0) or 0)
            drb  = float(stats.get("defensiveRebounds", 0) or 0)
            trb  = float(stats.get("rebounds", orb + drb) or 0)
            tov  = float(stats.get("turnovers", 0) or 0)
            ast  = float(stats.get("assists", 0) or 0)
            stl  = float(stats.get("steals", 0) or 0)
            blk  = float(stats.get("blocks", 0) or 0)
            pts  = float(stats.get("points", 0) or 0)
            pm   = float(stats.get("plusMinus", 0) or 0)

            player_info = p.get("player", {})
            player_name = player_info.get("name", "")
            player_code = str(player_info.get("id", ""))

            player_rows.append({
                "game_id":      game_id,
                "season":       season_label,
                "season_code":  season_code,
                "league":       league,
                "date":         game_date.isoformat(),
                "team_id":      team_id,
                "team_name":    team_name,
                "team_side":    side,
                "player_name":  player_name,
                "player_code":  player_code,
                "minutes":      round(minutes, 3),
                "points":       pts,
                "fga":          fga,
                "fgm":          fgm,
                "fta":          fta,
                "ftm":          ftm,
                "orb":          orb,
                "drb":          drb,
                "trb":          trb,
                "ast":          ast,
                "tov":          tov,
                "stl":          stl,
                "blk":          blk,
                "plus_minus":   pm,
                "valuation":    0.0,
                "starter":      not bool(p.get("substitute", True)),
                "team_points":  team_score,
                "game_minutes": float(game_minutes),
                "ot_periods":   ot_periods,
            })

    if not player_rows:
        return [], None

    sched_row = {
        "game_id":      game_id,
        "season":       season_label,
        "season_code":  season_code,
        "league":       league,
        "date":         game_date.isoformat(),
        "home_team":    home_name,
        "away_team":    away_name,
        "home_team_id": get_team_id(home_name),
        "away_team_id": get_team_id(away_name),
        "home_points":  home_score,
        "away_points":  away_score,
        "game_total":   home_score + away_score,
        "phase":        "RS",
        "round":        0,
        "played":       True,
    }
    return player_rows, sched_row


# ---------------------------------------------------------------------------
# Silver persistence
# ---------------------------------------------------------------------------

def _save_to_silver(league: str, schedule_rows: list[dict], player_rows: list[dict]) -> None:
    """Upsert new rows into silver parquets (dedup by game_id)."""
    if schedule_rows:
        sched_df = pd.DataFrame(schedule_rows).drop_duplicates("game_id")
        sched_df["date"] = pd.to_datetime(sched_df["date"])
        sched_df = sched_df.sort_values("date")

        path = settings.silver_dir / "schedules" / f"{league}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pd.read_parquet(path)
            sched_df = (
                pd.concat([existing, sched_df])
                .drop_duplicates("game_id")
                .sort_values("date")
                .reset_index(drop=True)
            )
        sched_df.to_parquet(path, index=False)
        typer.echo(f"  ✓ Schedule ({league}): {len(sched_df)} games → {path.name}")

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
        player_df["pace_per40"] = (
            player_df["possessions"] / player_df["game_minutes"].clip(lower=1)
        ) * 40
        player_df["poss_norm"] = player_df["possessions"] * ot_scale

        path = settings.silver_dir / "box_scores" / f"{league}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pd.read_parquet(path)
            dedup = [c for c in ["game_id", "player_code", "team_id"]
                     if c in player_df.columns and c in existing.columns]
            player_df = (
                pd.concat([existing, player_df])
                .drop_duplicates(subset=dedup)
                .reset_index(drop=True)
            )
        player_df.to_parquet(path, index=False)
        typer.echo(
            f"  ✓ Box scores ({league}): "
            f"{player_df['game_id'].nunique()} games, "
            f"{len(player_df)} player-rows → {path.name}"
        )
        seasons = sorted(player_df["season"].dropna().unique().tolist())
        typer.echo(f"    Seasons: {seasons}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    leagues: list[str] = typer.Option(
        ["acb", "bbl", "bsl", "nba", "lkl", "koris", "nbl_cz", "aba", "cba", "hung"],
        "--league",
        help="League(s): acb, bbl, bsl, nba, lkl, koris, nbl_cz, aba, cba, hung. Repeat for multiple.",
    ),
    seasons: list[str] = typer.Option(
        [],
        "--season",
        help="Season code(s) e.g. ACB2024, BBL2023. Defaults to all.",
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip already-cached bronze files",
    ),
    build_gold: bool = typer.Option(
        True,
        "--build-gold/--no-build-gold",
        help="Rebuild gold table after scraping",
    ),
    incremental: bool = typer.Option(
        False,
        "--incremental",
        help="Refresh current-season data only: invalidate schedule cache, keep existing box scores.",
    ),
) -> None:
    """Scrape ACB / BBL / BSL / NBA / LKL / KORIS / NBL_CZ / ABA / CBA / HUNG historical data and build silver/gold tables."""
    setup_logging(log_level="INFO", log_dir=settings.logs_dir)
    settings.ensure_dirs()

    if incremental:
        # Auto-target current (highest) seasons if none specified
        if not seasons:
            current: list[str] = []
            if "acb" in leagues:
                current.append(max(ACB_SEASON_IDS))
            for lk in leagues:
                if lk in SOFASCORE_LEAGUES:
                    current.append(max(SOFASCORE_LEAGUES[lk]["seasons"]))
            seasons[:] = current
            typer.echo(f"Incremental mode — targeting seasons: {seasons}")
        # Invalidate schedule caches so the API returns fresh event lists
        for lk in leagues:
            if lk == "acb":
                for sc in seasons:
                    if sc in ACB_SEASON_IDS:
                        p = settings.bronze_dir / "acb" / sc / f"calendar_{sc}.json"
                        if p.exists():
                            p.unlink()
                            typer.echo(f"  Invalidated: {p}")
            elif lk in SOFASCORE_LEAGUES:
                for sc in seasons:
                    if sc in SOFASCORE_LEAGUES[lk]["seasons"]:
                        p = settings.bronze_dir / lk / sc / f"schedule_{sc}.json"
                        if p.exists():
                            p.unlink()
                            typer.echo(f"  Invalidated: {p}")

    async def _run() -> None:
        async with httpx.AsyncClient(
            headers=HEADERS_BROWSER,
            follow_redirects=True,
            timeout=30,
        ) as acb_client, httpx.AsyncClient(
            headers=HEADERS_SOFASCORE,
            follow_redirects=True,
            timeout=30,
        ) as ss_client:

            for league_key in leagues:
                typer.echo(f"\n{'─'*60}")
                typer.echo(f"  {league_key.upper()}")
                typer.echo(f"{'─'*60}")

                if league_key == "acb":
                    all_sched: list[dict] = []
                    all_players: list[dict] = []

                    for season_code, (season_label, temporada_id) in ACB_SEASON_IDS.items():
                        if seasons and season_code not in seasons:
                            continue
                        typer.echo(f"\n  ── {season_label} ({season_code}) ──")
                        bronze_dir = settings.bronze_dir / "acb" / season_code
                        bronze_dir.mkdir(parents=True, exist_ok=True)

                        sched, players = await acb_scrape_season(
                            acb_client, season_code, season_label,
                            temporada_id, bronze_dir, skip_existing
                        )
                        all_sched.extend(sched)
                        all_players.extend(players)

                    if all_sched or all_players:
                        _save_to_silver("acb", all_sched, all_players)
                    else:
                        typer.echo("  ⚠  No ACB data collected")

                elif league_key in SOFASCORE_LEAGUES:
                    cfg = SOFASCORE_LEAGUES[league_key]
                    all_sched = []
                    all_players = []

                    for season_code, (season_label, season_id) in cfg["seasons"].items():
                        if seasons and season_code not in seasons:
                            continue
                        typer.echo(f"\n  ── {season_label} ({season_code}) ──")
                        bronze_dir = settings.bronze_dir / league_key / season_code
                        bronze_dir.mkdir(parents=True, exist_ok=True)

                        sched, players = await sofascore_scrape_season(
                            ss_client, league_key, cfg["tournament_id"],
                            season_code, season_label, season_id,
                            bronze_dir, skip_existing
                        )
                        all_sched.extend(sched)
                        all_players.extend(players)

                    if all_sched or all_players:
                        _save_to_silver(league_key, all_sched, all_players)
                    else:
                        typer.echo(f"  ⚠  No {league_key.upper()} data collected")

                else:
                    typer.echo(f"  Unknown league {league_key!r}. Valid: acb, bbl, bsl, nba, lkl, koris, nbl_cz, aba, cba, hung")

    asyncio.run(_run())

    if build_gold:
        typer.echo(f"\n{'─'*60}")
        typer.echo("  Building gold feature table …")
        typer.echo(f"{'─'*60}")
        from src.pipeline.silver_to_gold import build_gold_table
        gold_df = build_gold_table()
        if not gold_df.empty:
            typer.echo(f"  ✓ Gold: {len(gold_df)} matchup rows, {len(gold_df.columns)} features")
        else:
            typer.echo("  ⚠  Gold table empty — check silver data")

    typer.echo("\nDone.\n")


if __name__ == "__main__":
    app()
