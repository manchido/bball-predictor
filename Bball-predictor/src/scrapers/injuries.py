"""
Injury / Player Availability Scraper.

Tries to fetch current injury reports from RealGM.
Returns empty dict on any failure so the rest of the pipeline
can proceed without adjustment (graceful fallback).

Output format:
  {team_id: ["player_name_1", "player_name_2", ...]}  # unavailable players
"""

from __future__ import annotations

from typing import Any

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.utils.config import settings
from src.utils.ids import get_team_id
from src.utils.logging import logger

HEADERS = {
    "User-Agent": "BballPredictor/1.0 (research; +https://github.com/user/bball-predictor)",
    "Accept-Language": "en-US,en;q=0.9",
}

# RealGM injury report URLs per league
INJURY_URLS: dict[str, str] = {
    "euroleague": f"{settings.realgm_base_url}/international/injuries.html?league=EuroLeague",
    "eurocup": f"{settings.realgm_base_url}/international/injuries.html?league=EuroCup",
    "acb": f"{settings.realgm_base_url}/international/injuries.html?league=ACB",
    "bsl": f"{settings.realgm_base_url}/international/injuries.html?league=BSL",
    "bbl": f"{settings.realgm_base_url}/international/injuries.html?league=BBL",
}


class InjuryScraper:
    """Fetch and parse injury reports. Always falls back to empty dict."""

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=2, min=3, max=15),
        reraise=False,
    )
    async def fetch_today(self, league: str) -> dict[str, list[str]]:
        """
        Return {team_id: [unavailable_player_names]} for `league`.
        Returns {} on any scraping failure.
        """
        url = INJURY_URLS.get(league.lower())
        if not url:
            logger.info("No injury URL configured for league={}", league)
            return {}

        try:
            async with httpx.AsyncClient(headers=HEADERS, timeout=20) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                html = resp.text
            return _parse_injury_html(html)
        except Exception as exc:
            logger.warning(
                "Injury scrape failed for league={} ({}). Using empty fallback.", league, exc
            )
            return {}


def _parse_injury_html(html: str) -> dict[str, list[str]]:
    """Parse RealGM injury table → {team_id: [player_names]}."""
    soup = BeautifulSoup(html, "lxml")
    result: dict[str, list[str]] = {}

    table = soup.find("table")
    if table is None:
        return result

    headers = [th.get_text(strip=True).upper() for th in table.find_all("th")]

    def col_idx(name: str) -> int:
        try:
            return headers.index(name)
        except ValueError:
            return -1

    player_idx = col_idx("PLAYER") if col_idx("PLAYER") >= 0 else 0
    team_idx = col_idx("TEAM") if col_idx("TEAM") >= 0 else 1
    status_idx = col_idx("STATUS") if col_idx("STATUS") >= 0 else 3

    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) < 2:
            continue

        try:
            player_name = cols[player_idx].get_text(strip=True) if player_idx < len(cols) else ""
            team_raw = cols[team_idx].get_text(strip=True) if team_idx < len(cols) else ""
            status = cols[status_idx].get_text(strip=True).upper() if status_idx < len(cols) else ""

            # Only count players who are OUT or DOUBTFUL as unavailable
            if status in ("OUT", "DOUBTFUL", "DID NOT PLAY", "DNP"):
                team_id = get_team_id(team_raw)
                result.setdefault(team_id, []).append(player_name)
        except Exception as exc:
            logger.debug("Injury row parse error: {}", exc)

    logger.info("Parsed injury report: {} teams with unavailable players", len(result))
    return result
