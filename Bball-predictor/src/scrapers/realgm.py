"""
RealGM International Basketball Scraper.

Scrapes schedules, box scores, and team stats for EuroLeague,
EuroCup, and national leagues (ACB, BSL, BBL, etc.).

Rate limiting: 1 request per REALGM_REQUEST_DELAY seconds.
Retries: exponential backoff via tenacity (max 3 attempts).
Caching: 24-hour disk cache per URL.
robots.txt: checked once per domain on first run.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.robotparser
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.cache import DiskCache
from src.utils.config import settings
from src.utils.ids import make_game_id, get_team_id
from src.utils.logging import logger

# ---------------------------------------------------------------------------
# League → RealGM path mapping
# ---------------------------------------------------------------------------
LEAGUE_PATHS: dict[str, str] = {
    "euroleague": "/international/leagues/home.html?league=EuroLeague",
    "eurocup": "/international/leagues/home.html?league=EuroCup",
    "acb": "/international/leagues/home.html?league=ACB",
    "bsl": "/international/leagues/home.html?league=BSL",
    "bbl": "/international/leagues/home.html?league=BBL",
    "lega": "/international/leagues/home.html?league=LegaBasket",
    "pro-a": "/international/leagues/home.html?league=Pro-A",
}

SCHEDULE_PATH = "/international/leagues/schedule.html?league={league_key}&season_division_id={season_id}&type=Regular%20Season"
BOX_SCORE_BASE = "https://basketball.realgm.com"

HEADERS = {
    "User-Agent": "BballPredictor/1.0 (research; +https://github.com/user/bball-predictor)",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml",
}


class RealGMScraper:
    """Async scraper for RealGM international basketball data."""

    def __init__(self) -> None:
        self._base = settings.realgm_base_url
        self._delay = settings.realgm_request_delay
        self._cache = DiskCache(
            cache_dir=settings.bronze_dir / "realgm" / "_cache",
            ttl_seconds=86400,  # 24h
        )
        self._robots_allowed: Optional[bool] = None
        self._last_request: float = 0.0

    # ------------------------------------------------------------------
    # robots.txt compliance
    # ------------------------------------------------------------------
    def _check_robots(self) -> bool:
        if self._robots_allowed is not None:
            return self._robots_allowed
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"{self._base}/robots.txt")
        try:
            rp.read()
            self._robots_allowed = rp.can_fetch(HEADERS["User-Agent"], self._base)
        except Exception:
            # If robots.txt is unreachable, allow (conservative crawl anyway)
            self._robots_allowed = True
        logger.info("RealGM robots.txt — crawl allowed: {}", self._robots_allowed)
        return self._robots_allowed

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    async def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._delay:
            await asyncio.sleep(self._delay - elapsed)

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        reraise=True,
    )
    async def _get(self, url: str) -> str:
        """Throttled, cached, retrying GET returning HTML text."""
        cached = self._cache.get(url)
        if cached is not None:
            logger.debug("Cache HIT: {}", url)
            return cached

        if not self._check_robots():
            raise PermissionError(f"robots.txt disallows crawling {self._base}")

        await self._throttle()
        async with httpx.AsyncClient(headers=HEADERS, timeout=30) as client:
            logger.debug("GET {}", url)
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text

        self._last_request = time.monotonic()
        self._cache.set(url, html)
        return html

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def fetch_schedule(
        self,
        league: str,
        season: str,
        season_id: str = "",
    ) -> list[dict[str, Any]]:
        """
        Fetch and parse the schedule page for a league/season.

        Returns a list of game dicts:
          {game_id, date, home_team, away_team, home_team_id, away_team_id,
           league, season, game_url, status}
        """
        league_key = league.upper()
        url = (
            f"{self._base}/international/leagues/schedule.html"
            f"?league={league_key}&season_division_id={season_id}&type=Regular%20Season"
        )
        html = await self._get(url)

        games = _parse_schedule_html(html, league=league, season=season)

        # Save raw HTML to bronze
        raw_path = (
            settings.bronze_dir / "realgm" / "schedules"
            / f"{league}_{season}_{date.today().isoformat()}.html"
        )
        raw_path.write_text(html, encoding="utf-8")
        logger.info("Saved schedule HTML → {}", raw_path)

        return games

    async def fetch_box_score(self, game_url: str, game_id: str) -> dict[str, Any]:
        """
        Fetch and parse a box score page.

        Returns dict with keys:
          home_team_id, away_team_id, home_points, away_points,
          game_minutes, ot_periods, players: [{player_id, team_id,
          minutes, points, fga, orb, tov, fta, ...}]
        """
        url = game_url if game_url.startswith("http") else f"{BOX_SCORE_BASE}{game_url}"
        html = await self._get(url)

        box = _parse_box_score_html(html, game_id=game_id)

        raw_path = (
            settings.bronze_dir / "realgm" / "box_scores"
            / f"{game_id}.html"
        )
        raw_path.write_text(html, encoding="utf-8")
        logger.info("Saved box score HTML → {}", raw_path)

        return box

    async def fetch_team_stats(
        self, league: str, season: str, season_id: str = ""
    ) -> list[dict[str, Any]]:
        """
        Fetch team-level advanced stats table for a league/season.

        Returns list of team stat dicts (one per team).
        """
        league_key = league.upper()
        url = (
            f"{self._base}/international/leagues/team-stats.html"
            f"?league={league_key}&season_division_id={season_id}"
            f"&stat_type=Advanced&stage=Regular+Season"
        )
        html = await self._get(url)
        stats = _parse_team_stats_html(html, league=league, season=season)

        raw_path = (
            settings.bronze_dir / "realgm" / "team_stats"
            / f"{league}_{season}_advanced.html"
        )
        raw_path.write_text(html, encoding="utf-8")
        logger.info("Saved team stats HTML → {}", raw_path)

        return stats

    async def fetch_today_schedule(self, league: str) -> list[dict[str, Any]]:
        """Convenience: fetch today's scheduled games for a league."""
        today = date.today().isoformat()
        url = f"{self._base}/international/leagues/schedule.html?league={league.upper()}&date={today}"
        html = await self._get(url)
        return _parse_schedule_html(html, league=league, season="current")


# ---------------------------------------------------------------------------
# HTML parsers
# ---------------------------------------------------------------------------

def _parse_schedule_html(html: str, league: str, season: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    games: list[dict[str, Any]] = []

    # RealGM schedule tables vary by league — try common patterns
    table = soup.find("table", {"class": lambda c: c and "tablesaw" in c})
    if table is None:
        table = soup.find("table")
    if table is None:
        logger.warning("No schedule table found for league={} season={}", league, season)
        return games

    rows = table.find_all("tr")
    current_date: Optional[date] = None

    for row in rows:
        # Date header rows
        if row.find("th") and not row.find("td"):
            date_text = row.get_text(strip=True)
            try:
                current_date = datetime.strptime(date_text, "%B %d, %Y").date()
            except ValueError:
                pass
            continue

        cols = row.find_all("td")
        if len(cols) < 3:
            continue

        try:
            away_cell = cols[0]
            home_cell = cols[2]
            score_cell = cols[1] if len(cols) > 1 else None

            away_raw = away_cell.get_text(strip=True)
            home_raw = home_cell.get_text(strip=True)

            if not away_raw or not home_raw:
                continue

            # Game link
            link_tag = score_cell.find("a") if score_cell else None
            game_url = link_tag["href"] if link_tag and link_tag.get("href") else ""

            game_date = current_date or date.today()
            game_id = make_game_id(game_date, home_raw, away_raw)

            games.append({
                "game_id": game_id,
                "date": game_date.isoformat(),
                "home_team": home_raw,
                "away_team": away_raw,
                "home_team_id": get_team_id(home_raw),
                "away_team_id": get_team_id(away_raw),
                "league": league,
                "season": season,
                "game_url": game_url,
                "status": "scheduled" if not game_url else "final",
            })
        except Exception as exc:
            logger.debug("Schedule row parse error: {}", exc)
            continue

    logger.info("Parsed {} games from schedule ({} {})", len(games), league, season)
    return games


def _parse_box_score_html(html: str, game_id: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    result: dict[str, Any] = {
        "game_id": game_id,
        "home_team_id": "",
        "away_team_id": "",
        "home_points": 0,
        "away_points": 0,
        "game_minutes": 40,
        "ot_periods": 0,
        "players": [],
    }

    # Detect OT — look for OT columns in score summary
    score_table = soup.find("table", {"id": lambda i: i and "score" in i.lower()})
    if score_table:
        headers = [th.get_text(strip=True) for th in score_table.find_all("th")]
        ot_count = sum(1 for h in headers if h.upper().startswith("OT"))
        result["ot_periods"] = ot_count
        result["game_minutes"] = 40 + ot_count * 5

    # Parse player box score tables
    tables = soup.find_all("table", {"class": lambda c: c and "tablesaw" in str(c)})

    team_toggle = 0  # 0=away, 1=home — RealGM shows away first
    for table in tables:
        team_key = "away" if team_toggle == 0 else "home"
        team_toggle = 1 - team_toggle

        # Try to extract team name from section heading
        heading = table.find_previous("h2") or table.find_previous("h3")
        team_raw = heading.get_text(strip=True) if heading else ""
        team_id = get_team_id(team_raw) if team_raw else f"{game_id}_{team_key}"

        if team_key == "away":
            result["away_team_id"] = team_id
        else:
            result["home_team_id"] = team_id

        col_headers = [th.get_text(strip=True).upper() for th in table.find_all("th")]

        def col_idx(name: str) -> int:
            try:
                return col_headers.index(name)
            except ValueError:
                return -1

        pts_idx = col_idx("PTS")
        min_idx = col_idx("MIN")
        fga_idx = col_idx("FGA")
        orb_idx = col_idx("ORB") if col_idx("ORB") >= 0 else col_idx("OREB")
        tov_idx = col_idx("TOV") if col_idx("TOV") >= 0 else col_idx("TO")
        fta_idx = col_idx("FTA")

        team_points = 0
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if not cols:
                continue

            def safe_float(idx: int) -> float:
                if idx < 0 or idx >= len(cols):
                    return 0.0
                txt = cols[idx].get_text(strip=True).replace(":", ".")
                try:
                    return float(txt)
                except ValueError:
                    return 0.0

            # Minutes: convert MM:SS → decimal
            min_raw = cols[min_idx].get_text(strip=True) if min_idx >= 0 else "0"
            try:
                if ":" in min_raw:
                    m, s = min_raw.split(":")
                    minutes = int(m) + int(s) / 60
                else:
                    minutes = float(min_raw)
            except (ValueError, IndexError):
                minutes = 0.0

            pts = safe_float(pts_idx)
            team_points += pts

            player_name_cell = cols[0]
            player_raw = player_name_cell.get_text(strip=True)

            result["players"].append({
                "game_id": game_id,
                "team_id": team_id,
                "team_side": team_key,
                "player_name": player_raw,
                "minutes": minutes,
                "points": pts,
                "fga": safe_float(fga_idx),
                "orb": safe_float(orb_idx),
                "tov": safe_float(tov_idx),
                "fta": safe_float(fta_idx),
            })

        if team_key == "away":
            result["away_points"] = team_points
        else:
            result["home_points"] = team_points

    return result


def _parse_team_stats_html(html: str, league: str, season: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    stats_list: list[dict[str, Any]] = []

    table = soup.find("table")
    if table is None:
        return stats_list

    headers = [th.get_text(strip=True).upper() for th in table.find_all("th")]

    def col_idx(name: str) -> int:
        try:
            return headers.index(name)
        except ValueError:
            return -1

    team_idx = col_idx("TEAM") if col_idx("TEAM") >= 0 else 0

    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) < 3:
            continue

        def val(name: str) -> float:
            idx = col_idx(name)
            if idx < 0 or idx >= len(cols):
                return 0.0
            try:
                return float(cols[idx].get_text(strip=True))
            except ValueError:
                return 0.0

        team_raw = cols[team_idx].get_text(strip=True)
        stats_list.append({
            "team_id": get_team_id(team_raw),
            "team_name": team_raw,
            "league": league,
            "season": season,
            "gp": val("GP"),
            "off_rtg": val("ORTG"),
            "def_rtg": val("DRTG"),
            "net_rtg": val("NET"),
            "pace": val("PACE"),
            "efg_pct": val("EFG%"),
            "ts_pct": val("TS%"),
            "tov_pct": val("TOV%"),
            "orb_pct": val("ORB%"),
        })

    logger.info("Parsed {} team stat rows ({} {})", len(stats_list), league, season)
    return stats_list
