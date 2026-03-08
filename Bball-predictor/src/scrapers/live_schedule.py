"""
Live schedule fetcher — today's upcoming games from official per-league APIs.

Sources
───────
  EuroLeague / EuroCup → api-live.euroleague.net/v2  (official, no auth)
  ACB / BBL / BSL / NBA and more → api.sofascore.com (unofficial, browser headers)

Each league returns the same game-dict schema used by _predict_games:
  {game_id, date, home_team, away_team, home_team_id, away_team_id,
   league, season, status}
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from zoneinfo import ZoneInfo

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.utils.ids import make_game_id, get_team_id
from src.utils.logging import logger

_UTC = ZoneInfo("UTC")

EL_BASE  = "https://api-live.euroleague.net/v2"
SF_BASE  = "https://api.sofascore.com/api/v1"
ESPN_NBA = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

HEADERS_EL = {
    "Accept": "application/json",
    "User-Agent": "BballPredictor/1.0 (research)",
}
HEADERS_SF = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.sofascore.com/",
}
HEADERS_ESPN = {
    "User-Agent": "BballPredictor/1.0 (research)",
    "Accept": "application/json",
}

# Current season codes — update each October
_EL_CURRENT: dict[str, tuple[str, str, str]] = {
    # league → (competition_code, season_code, season_label)
    "euroleague": ("E", "E2025", "2025-26"),
    "eurocup":    ("U", "U2025", "2025-26"),
}

_SF_CURRENT: dict[str, tuple[int, int, str]] = {
    # league → (tournament_id, season_id, season_label)
    "acb":    (264,   80922, "2025-26"),
    "bbl":    (227,   79994, "2025-26"),
    "bsl":    (519,   81036, "2025-26"),
    "nba":    (132,   80229, "2025-26"),
    "lkl":    (975,   80356, "2025-26"),
    "koris":  (226,   79938, "2025-26"),
    "nbl_cz": (250,   77912, "2025-26"),
    "aba":    (235,   80150, "2025-26"),
    "cba":    (1566,  85375, "2025-26"),
    "hung":   (10594, 79941, "2025-26"),
}


def _is_retryable(exc: BaseException) -> bool:
    """Retry on transient server/network errors only — never on 4xx client errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500  # 5xx = server-side transient
    return isinstance(exc, httpx.TimeoutException)  # network timeout


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=15),
    reraise=True,
)
async def _get_json(client: httpx.AsyncClient, url: str, **kwargs) -> dict | list:
    resp = await client.get(url, timeout=20, **kwargs)
    resp.raise_for_status()
    return resp.json()


class LiveScheduleFetcher:
    """Fetch scheduled games for any date using official per-league APIs."""

    async def fetch_today(self, league: str, target_date: date | None = None) -> list[dict]:
        """Return game-dicts for `target_date` (defaults to today) in `league`."""
        target_date = target_date or date.today()
        try:
            if league in _EL_CURRENT:
                return await self._fetch_el_today(league, target_date)
            if league in _SF_CURRENT:
                return await self._fetch_sf_today(league, target_date)
        except Exception:
            logger.warning("LiveScheduleFetcher primary source failed for league={}, trying fallbacks", league)
            # ESPN fallback for NBA when SofaScore is rate-limited
            if league == "nba":
                try:
                    return await self._fetch_espn_nba_today(target_date)
                except Exception:
                    logger.exception("ESPN NBA fallback also failed")
        return []

    # ------------------------------------------------------------------
    # EuroLeague / EuroCup
    # ------------------------------------------------------------------
    async def _fetch_el_today(self, league: str, today: date) -> list[dict]:
        comp, season_code, season_label = _EL_CURRENT[league]
        url = f"{EL_BASE}/competitions/{comp}/seasons/{season_code}/games"

        async with httpx.AsyncClient(headers=HEADERS_EL, timeout=20) as c:
            # Fetch all season games — status=None games are missed by gameStatus filter
            # Date filtering is done locally below
            data = await _get_json(c, url, params={"limit": 500})
        games_raw = data.get("data", data) if isinstance(data, dict) else data

        results: list[dict] = []
        for g in games_raw if isinstance(games_raw, list) else []:
            raw_date = g.get("utcDate") or g.get("date") or ""
            try:
                gdate = datetime.fromisoformat(raw_date.replace("Z", "+00:00")).astimezone(_UTC).date()
            except Exception:
                continue
            if gdate != today:
                continue

            home_raw = (g.get("local") or {}).get("club", {}).get("name", "")
            away_raw = (g.get("road")  or {}).get("club", {}).get("name", "")
            if not home_raw or not away_raw:
                continue

            results.append(_build_game_dict(today, home_raw, away_raw, league, season_label))

        logger.info("LiveScheduleFetcher [{}]: {} games today", league, len(results))
        return results

    # ------------------------------------------------------------------
    # SofaScore (BBL / BSL / NBA / LKL / KORIS / NBL_CZ / ABA / CBA / HUNG)
    # ------------------------------------------------------------------
    async def _fetch_sf_today(self, league: str, today: date) -> list[dict]:
        """
        Use the date-based scheduled-events endpoint so we catch games that
        have already started (live) or finished today, not just upcoming ones.
        Falls back to the per-season /events/next paginator if the date
        endpoint returns 404 (older SofaScore API versions).
        """
        t_id, s_id, season_label = _SF_CURRENT[league]
        date_url = f"{SF_BASE}/sport/basketball/scheduled-events/{today.isoformat()}"

        async with httpx.AsyncClient(headers=HEADERS_SF, timeout=20) as c:
            try:
                data = await _get_json(c, date_url)
                events = data.get("events", [])
                results = []
                for e in events:
                    if e.get("tournament", {}).get("uniqueTournament", {}).get("id") != t_id:
                        continue
                    if not e.get("homeTeam", {}).get("name") or not e.get("awayTeam", {}).get("name"):
                        continue
                    # SofaScore's date endpoint includes events from surrounding days;
                    # verify the actual start timestamp falls on the requested date.
                    ts = e.get("startTimestamp", 0)
                    if ts and datetime.fromtimestamp(ts).date() != today:
                        continue
                    results.append(
                        _build_game_dict(today, e["homeTeam"]["name"], e["awayTeam"]["name"], league, season_label)
                    )
                logger.info("LiveScheduleFetcher [{}]: {} games today (date endpoint)", league, len(results))
                return results
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 404:
                    raise
                # Fall through to paginator if date endpoint unavailable

        # Fallback: paginate /events/next (misses already-started games)
        today_start = int(datetime.combine(today, datetime.min.time(), tzinfo=_UTC).timestamp())
        today_end   = today_start + 86_400

        results: list[dict] = []
        async with httpx.AsyncClient(headers=HEADERS_SF, timeout=20) as c:
            for page in range(5):
                url = f"{SF_BASE}/unique-tournament/{t_id}/season/{s_id}/events/next/{page}"
                try:
                    data = await _get_json(c, url)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        break
                    raise
                events = data.get("events", [])
                if not events:
                    break

                for ev in events:
                    ts = ev.get("startTimestamp", 0)
                    if ts >= today_end:
                        logger.info("LiveScheduleFetcher [{}]: {} games today (fallback paginator)", league, len(results))
                        return results
                    if today_start <= ts < today_end:
                        home_raw = ev.get("homeTeam", {}).get("name", "")
                        away_raw = ev.get("awayTeam", {}).get("name", "")
                        if home_raw and away_raw:
                            results.append(
                                _build_game_dict(today, home_raw, away_raw, league, season_label)
                            )

                await asyncio.sleep(0.5)

        logger.info("LiveScheduleFetcher [{}]: {} games today (fallback paginator)", league, len(results))
        return results

    # ------------------------------------------------------------------
    # ESPN (NBA fallback when SofaScore is rate-limited)
    # ------------------------------------------------------------------
    async def _fetch_espn_nba_today(self, today: date) -> list[dict]:
        """Fetch today's NBA games from ESPN's public scoreboard API (no auth needed)."""
        season_label = "2025-26"
        async with httpx.AsyncClient(headers=HEADERS_ESPN, timeout=15) as c:
            data = await _get_json(c, ESPN_NBA, params={"dates": today.strftime("%Y%m%d")})

        results: list[dict] = []
        for ev in data.get("events", []):
            for comp in ev.get("competitions", []):
                competitors = comp.get("competitors", [])
                home = next((t for t in competitors if t.get("homeAway") == "home"), None)
                away = next((t for t in competitors if t.get("homeAway") == "away"), None)
                if home and away:
                    home_name = home.get("team", {}).get("displayName", "")
                    away_name = away.get("team", {}).get("displayName", "")
                    if home_name and away_name:
                        results.append(_build_game_dict(today, home_name, away_name, "nba", season_label))

        logger.info("LiveScheduleFetcher [nba/ESPN fallback]: {} games today", len(results))
        return results

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _build_game_dict(
    game_date: date,
    home_raw: str,
    away_raw: str,
    league: str,
    season: str,
) -> dict:
    return {
        "game_id":      make_game_id(game_date, home_raw, away_raw),
        "date":         game_date.isoformat(),
        "home_team":    home_raw,
        "away_team":    away_raw,
        "home_team_id": get_team_id(home_raw),
        "away_team_id": get_team_id(away_raw),
        "league":       league,
        "season":       season,
        "status":       "scheduled",
    }
