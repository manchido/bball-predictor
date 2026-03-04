"""
Bookmaker Odds Scraper.

Primary source:   The Odds API (the-odds-api.com)
  — Legitimate REST API that aggregates bet365, Pinnacle, William Hill,
    and others. Bet365 data is accessible via this legal channel.
  — Free tier: 500 req/month. API key required (set ODDS_API_KEY in .env).

Fallback source:  OddsPortal HTML scrape
  — Used when The Odds API key is absent or quota exhausted.
  — Source is clearly stamped in every record as odds_source="oddsportal".

All timestamps are stored in America/St_Johns (UTC-3:30).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.utils.cache import DiskCache
from src.utils.config import settings
from src.utils.ids import make_game_id, get_team_id
from src.utils.logging import logger

_TZ = ZoneInfo("America/St_Johns")

# ---------------------------------------------------------------------------
# League → The Odds API sport key mapping
# ---------------------------------------------------------------------------
ODDS_API_SPORT_KEYS: dict[str, str] = {
    "euroleague": "basketball_euroleague",
    "eurocup": "basketball_eurocup",
    "acb": "basketball_spain_acb",
    "bsl": "basketball_turkey_bsl",
    "bbl": "basketball_germany_bbl",
    "lega": "basketball_italy_lega",
    "pro-a": "basketball_france_pro_a",
}

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDSPORTAL_BASE = "https://www.oddsportal.com"

# ---------------------------------------------------------------------------
# football.com / sportybet factsCenter API
# Same Sporty Group backend — football.com has no AWS WAF, prefer it.
# EuroLeague is the only target league covered (sr:tournament:138).
# Market 18 = main Over/Under total (half-point lines, decimal odds).
# ---------------------------------------------------------------------------
FOOTBALLCOM_API = "https://www.football.com/api/ng/factsCenter"
FOOTBALLCOM_EL_TOURNAMENT = "sr:tournament:138"   # EuroLeague
FOOTBALLCOM_TOTAL_MARKET_ID = "18"
FOOTBALLCOM_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.football.com/",
}

ODDSPORTAL_LEAGUE_PATHS: dict[str, str] = {
    "euroleague": "/basketball/europe/euroleague",
    "eurocup": "/basketball/europe/7days-eurocup",
    "acb": "/basketball/spain/acb",
    "bsl": "/basketball/turkey/bsl",
    "bbl": "/basketball/germany/bbl",
}


class OddsRecord:
    """Structured odds data for a single game."""

    __slots__ = (
        "game_id", "league", "match", "date",
        "book_total", "book_spread",
        "odds_source", "timestamp",
    )

    def __init__(
        self,
        game_id: str,
        league: str,
        match: str,
        date: str,
        book_total: float,
        book_spread: float,
        odds_source: str,
        timestamp: str,
    ) -> None:
        for k, v in zip(self.__slots__, (game_id, league, match, date, book_total, book_spread, odds_source, timestamp)):
            object.__setattr__(self, k, v)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


class OddsClient:
    """
    Fetches bookmaker totals + spreads.

    Priority: The Odds API → OddsPortal fallback.
    """

    def __init__(self) -> None:
        self._api_key = settings.odds_api_key
        self._cache = DiskCache(
            cache_dir=settings.bronze_dir / "odds" / "_cache",
            ttl_seconds=3600,  # 1h — odds move; don't cache too long
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    async def fetch_today_odds(self, league: str) -> dict[str, OddsRecord]:
        """
        Return {game_id: OddsRecord} for today's games in `league`.

        Priority:
          1. The Odds API  (EuroLeague only — other leagues not in catalog)
          2. football.com factsCenter API  (EuroLeague only — no auth required)
          3. OddsPortal HTML fallback  (all leagues, best-effort)
        """
        # -- 1. The Odds API (EuroLeague only) ----------------------------
        if self._api_key and league == "euroleague":
            try:
                records = await self._fetch_odds_api(league)
                if records:
                    self._save_raw(records, league, source="the-odds-api")
                    return {r.game_id: r for r in records}
            except Exception as exc:
                logger.warning("The Odds API failed ({}), trying football.com", exc)

        # -- 2. football.com factsCenter (EuroLeague only) ----------------
        if league == "euroleague":
            try:
                records = await self._fetch_footballcom(league)
                if records:
                    self._save_raw(records, league, source="football.com")
                    return {r.game_id: r for r in records}
            except Exception as exc:
                logger.warning("football.com odds failed ({}), trying OddsPortal", exc)

        # -- 3. OddsPortal HTML fallback ----------------------------------
        logger.info("Using OddsPortal fallback for league={}", league)
        records = await self._fetch_oddsportal(league)
        if records:
            self._save_raw(records, league, source="oddsportal")
        return {r.game_id: r for r in records}

    # ------------------------------------------------------------------
    # The Odds API
    # ------------------------------------------------------------------
    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=20),
        reraise=True,
    )
    async def _fetch_odds_api(self, league: str) -> list[OddsRecord]:
        sport_key = ODDS_API_SPORT_KEYS.get(league.lower())
        if not sport_key:
            logger.warning("No Odds API sport key for league={}", league)
            return []

        url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
        params = {
            "apiKey": self._api_key,
            "regions": "eu,uk",
            "markets": "totals,spreads",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }

        cache_key = f"odds-api:{sport_key}:{datetime.now(_TZ).date()}"
        cached = self._cache.get(cache_key)
        if cached:
            return [OddsRecord(**r) for r in cached]

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        remaining = resp.headers.get("x-requests-remaining", "?")
        logger.info("Odds API request OK — remaining quota: {}", remaining)

        records = _parse_odds_api_response(data, league)
        self._cache.set(cache_key, [r.to_dict() for r in records])
        return records

    # ------------------------------------------------------------------
    # football.com factsCenter API (EuroLeague totals — no auth)
    # ------------------------------------------------------------------
    async def _fetch_footballcom(self, league: str) -> list[OddsRecord]:
        """
        Fetch EuroLeague totals from football.com factsCenter API.

        Flow:
          1. GET sportList → locate EuroLeague events (sr:tournament:138)
          2. For each event, GET event?eventId=... → find market 18 (O/U total)
          3. Match home/away team to canonical names via get_team_id
        """
        if league != "euroleague":
            return []

        cache_key = f"footballcom:{league}:{datetime.now(_TZ).date()}"
        cached = self._cache.get(cache_key)
        if cached:
            return [OddsRecord(**r) for r in cached]

        today = datetime.now(_TZ).date()
        records: list[OddsRecord] = []

        async with httpx.AsyncClient(headers=FOOTBALLCOM_HEADERS, timeout=20) as client:
            # Step 1: get sport list to find EuroLeague event IDs
            resp = await client.get(f"{FOOTBALLCOM_API}/sportList")
            resp.raise_for_status()
            sport_data = resp.json()

            # Navigate: data (list of sports) → categories → tournaments
            # NOTE: sportList returns tournament-level info only (no event IDs).
            # This integration is currently a no-op; event IDs cannot be obtained
            # from this endpoint. Falls through to OddsPortal silently.
            el_event_ids: list[str] = []
            for sport in sport_data.get("data") or []:
                if sport.get("id") != "sr:sport:2":   # basketball
                    continue
                for cat in sport.get("categories") or []:
                    for tourn in cat.get("tournaments") or []:
                        if tourn.get("id") == FOOTBALLCOM_EL_TOURNAMENT:
                            for evt in tourn.get("events") or []:
                                eid = evt.get("id") or evt.get("eventId")
                                if eid:
                                    el_event_ids.append(str(eid))

            if not el_event_ids:
                logger.info("football.com: no EuroLeague events found in sportList")
                return []

            logger.info("football.com: {} EuroLeague events found", len(el_event_ids))

            # Step 2: fetch each event's markets
            for event_id in el_event_ids:
                try:
                    await asyncio.sleep(0.4)
                    resp = await client.get(
                        f"{FOOTBALLCOM_API}/event", params={"eventId": event_id}
                    )
                    resp.raise_for_status()
                    evt = resp.json().get("data") or resp.json()

                    home_raw = evt.get("homeTeamName") or ""
                    away_raw = evt.get("awayTeamName") or ""
                    commence_ts = evt.get("estimateStartTime") or evt.get("startTime") or 0

                    if not home_raw or not away_raw:
                        continue

                    game_dt = datetime.fromtimestamp(commence_ts / 1000, tz=_TZ) if commence_ts > 1e9 else None
                    game_date = game_dt.date() if game_dt else today
                    if game_date != today:
                        continue

                    # Find market 18 (main Over/Under)
                    book_total = 0.0
                    for market in evt.get("markets") or []:
                        if str(market.get("id")) == FOOTBALLCOM_TOTAL_MARKET_ID:
                            for outcome in market.get("outcomes") or []:
                                if outcome.get("name", "").lower() in ("over", "1"):
                                    book_total = float(outcome.get("total") or outcome.get("hc") or 0)
                                    break
                            if book_total:
                                break

                    if not book_total:
                        continue

                    game_id = make_game_id(game_date, home_raw, away_raw)
                    records.append(OddsRecord(
                        game_id=game_id,
                        league=league,
                        match=f"{away_raw} @ {home_raw}",
                        date=game_date.isoformat(),
                        book_total=book_total,
                        book_spread=0.0,
                        odds_source="football.com",
                        timestamp=_now_st_johns(),
                    ))

                except Exception as exc:
                    logger.debug("football.com event {} parse error: {}", event_id, exc)

        logger.info("football.com: {} EuroLeague odds records parsed", len(records))
        if records:
            self._cache.set(cache_key, [r.to_dict() for r in records])
        return records

    # ------------------------------------------------------------------
    # OddsPortal fallback (HTML scrape)
    # ------------------------------------------------------------------
    async def _fetch_oddsportal(self, league: str) -> list[OddsRecord]:
        path = ODDSPORTAL_LEAGUE_PATHS.get(league.lower())
        if not path:
            logger.warning("No OddsPortal path for league={}", league)
            return []

        url = f"{ODDSPORTAL_BASE}{path}/"
        cache_key = f"oddsportal:{league}:{datetime.now(_TZ).date()}"
        cached = self._cache.get(cache_key)
        if cached:
            return [OddsRecord(**r) for r in cached]

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; BballPredictor/1.0)",
            "Accept-Language": "en-US,en;q=0.9",
        }
        async with httpx.AsyncClient(headers=headers, timeout=30, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text

        records = _parse_oddsportal_html(html, league)
        self._cache.set(cache_key, [r.to_dict() for r in records])
        return records

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save_raw(self, records: list[OddsRecord], league: str, source: str) -> None:
        today = datetime.now(_TZ).date().isoformat()
        raw_path = (
            settings.bronze_dir / "odds"
            / f"{league}_{today}_{source}.json"
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            json.dumps([r.to_dict() for r in records], indent=2),
            encoding="utf-8",
        )
        logger.info("Saved {} odds records → {}", len(records), raw_path)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _now_st_johns() -> str:
    return datetime.now(_TZ).isoformat()


def _parse_odds_api_response(data: list[dict], league: str) -> list[OddsRecord]:
    records: list[OddsRecord] = []
    today = datetime.now(_TZ).date()

    for event in data:
        try:
            home_raw = event["home_team"]
            away_raw = event["away_team"]
            commence_ts = event.get("commence_time", "")

            # Filter to today's games only
            if commence_ts:
                game_dt = datetime.fromisoformat(commence_ts.replace("Z", "+00:00"))
                game_date = game_dt.astimezone(_TZ).date()
                if game_date != today:
                    continue
            else:
                game_date = today

            game_id = make_game_id(game_date, home_raw, away_raw)
            match_str = f"{away_raw} @ {home_raw}"

            book_total = 0.0
            book_spread = 0.0

            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market["key"] == "totals" and book_total == 0.0:
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == "Over":
                                book_total = float(outcome.get("point", 0))
                                break
                    elif market["key"] == "spreads" and book_spread == 0.0:
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == home_raw:
                                book_spread = float(outcome.get("point", 0))
                                break

            if book_total == 0.0:
                continue  # No total available — skip

            records.append(OddsRecord(
                game_id=game_id,
                league=league,
                match=match_str,
                date=game_date.isoformat(),
                book_total=book_total,
                book_spread=book_spread,
                odds_source="the-odds-api",
                timestamp=_now_st_johns(),
            ))

        except Exception as exc:
            logger.debug("Odds API event parse error: {}", exc)

    logger.info("Parsed {} odds records from The Odds API ({})", len(records), league)
    return records


def _parse_oddsportal_html(html: str, league: str) -> list[OddsRecord]:
    """
    Parse OddsPortal event listing page.

    OddsPortal serves totals in table rows. This is a best-effort
    parser — layout changes may require updates.
    Source is stamped as 'oddsportal' in all returned records.
    """
    soup = BeautifulSoup(html, "lxml")
    records: list[OddsRecord] = []
    today = datetime.now(_TZ).date()

    # OddsPortal event rows
    event_rows = soup.select("div.eventRow") or soup.select("tr.deactivate")

    for row in event_rows:
        try:
            # Team names
            teams = row.select("a.participant-name") or row.select(".names a")
            if len(teams) < 2:
                continue
            home_raw = teams[1].get_text(strip=True)
            away_raw = teams[0].get_text(strip=True)

            # Totals (look for O/U column)
            total_cell = row.select_one(".odds-wrap") or row.select_one("td.right")
            book_total = 0.0
            if total_cell:
                txt = total_cell.get_text(strip=True)
                # Pattern: "O 145.5" or just "145.5"
                import re
                m = re.search(r"(\d{2,3}\.?\d*)", txt)
                if m:
                    book_total = float(m.group(1))

            if book_total == 0.0:
                continue

            game_id = make_game_id(today, home_raw, away_raw)
            records.append(OddsRecord(
                game_id=game_id,
                league=league,
                match=f"{away_raw} @ {home_raw}",
                date=today.isoformat(),
                book_total=book_total,
                book_spread=0.0,  # OddsPortal spread parsing optional
                odds_source="oddsportal",
                timestamp=_now_st_johns(),
            ))

        except Exception as exc:
            logger.debug("OddsPortal row parse error: {}", exc)

    logger.info("Parsed {} odds records from OddsPortal ({})", len(records), league)
    return records
