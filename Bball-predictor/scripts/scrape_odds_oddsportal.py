"""
Scrape historical over/under closing lines from OddsPortal.

Covers configured leagues × 3 seasons. Results are cached to bronze JSON
and then written to data/silver/odds.parquet for automatic inclusion
in the gold feature table (book_total column).

Two-phase scraping strategy:
  Phase 1 (fast):     switch results table to O/U market view, extract inline odds
                      per game row — avoids visiting individual game pages.
  Phase 2 (fallback): for games without inline O/U, visit individual game pages.
                      Skipped when --no-game-pages is passed.

Usage:
    python scripts/scrape_odds_oddsportal.py                           # all leagues/seasons
    python scripts/scrape_odds_oddsportal.py --league nba              # one league, all seasons
    python scripts/scrape_odds_oddsportal.py --league nba --season 2024-25
    python scripts/scrape_odds_oddsportal.py --no-game-pages           # fast mode: inline only
    python scripts/scrape_odds_oddsportal.py --build-silver-only       # skip scraping, rebuild parquet
    python scripts/scrape_odds_oddsportal.py --dry-run                 # show targets without scraping
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import typer

# ---------------------------------------------------------------------------
# League → OddsPortal results URL mapping
# ---------------------------------------------------------------------------
# NOTE: OddsPortal URL slugs for historical seasons use the pattern:
#   /basketball/{country}/{league}-{year1}-{year2}/results/
# Current (live) season uses just:
#   /basketball/{country}/{league}/results/
#
# BSL note: OddsPortal lists Turkish basketball as "super-lig" (same slug
#   as football in other countries — confirmed working for basketball/turkey/).
# LKL / KORIS / NBL_CZ / ABA / HUNG: bookmakers rarely post O/U for these;
#   Phase 1 inline extraction will quickly reveal empty leagues.
# ---------------------------------------------------------------------------
LEAGUE_SEASON_URLS: dict[str, dict[str, str]] = {
    # ── European top competitions ────────────────────────────────────────────
    "euroleague": {
        "2023-24": "https://www.oddsportal.com/basketball/europe/euroleague-2023-2024/results/",
        "2024-25": "https://www.oddsportal.com/basketball/europe/euroleague-2024-2025/results/",
        "2025-26": "https://www.oddsportal.com/basketball/europe/euroleague/results/",
    },
    "eurocup": {
        "2023-24": "https://www.oddsportal.com/basketball/europe/eurocup-2023-2024/results/",
        "2024-25": "https://www.oddsportal.com/basketball/europe/eurocup-2024-2025/results/",
        "2025-26": "https://www.oddsportal.com/basketball/europe/eurocup/results/",
    },
    "aba": {
        "2023-24": "https://www.oddsportal.com/basketball/europe/aba-league-2023-2024/results/",
        "2024-25": "https://www.oddsportal.com/basketball/europe/aba-league-2024-2025/results/",
        "2025-26": "https://www.oddsportal.com/basketball/europe/aba-league/results/",
    },
    # ── National leagues (good O/U coverage) ────────────────────────────────
    "nba": {
        "2023-24": "https://www.oddsportal.com/basketball/usa/nba-2023-2024/results/",
        "2024-25": "https://www.oddsportal.com/basketball/usa/nba-2024-2025/results/",
        "2025-26": "https://www.oddsportal.com/basketball/usa/nba/results/",
    },
    "acb": {
        "2023-24": "https://www.oddsportal.com/basketball/spain/acb-2023-2024/results/",
        "2024-25": "https://www.oddsportal.com/basketball/spain/acb-2024-2025/results/",
        "2025-26": "https://www.oddsportal.com/basketball/spain/acb/results/",
    },
    "bbl": {
        "2023-24": "https://www.oddsportal.com/basketball/germany/bbl-2023-2024/results/",
        "2024-25": "https://www.oddsportal.com/basketball/germany/bbl-2024-2025/results/",
        "2025-26": "https://www.oddsportal.com/basketball/germany/bbl/results/",
    },
    "bsl": {
        "2023-24": "https://www.oddsportal.com/basketball/turkey/super-lig-2023-2024/results/",
        "2024-25": "https://www.oddsportal.com/basketball/turkey/super-lig-2024-2025/results/",
        "2025-26": "https://www.oddsportal.com/basketball/turkey/super-lig/results/",
    },
    # ── National leagues (sparse/variable O/U coverage) ─────────────────────
    "cba": {
        "2023-24": "https://www.oddsportal.com/basketball/china/cba-2023-2024/results/",
        "2024-25": "https://www.oddsportal.com/basketball/china/cba-2024-2025/results/",
        "2025-26": "https://www.oddsportal.com/basketball/china/cba/results/",
    },
    "lkl": {
        "2023-24": "https://www.oddsportal.com/basketball/lithuania/lkl-2023-2024/results/",
        "2024-25": "https://www.oddsportal.com/basketball/lithuania/lkl-2024-2025/results/",
        "2025-26": "https://www.oddsportal.com/basketball/lithuania/lkl/results/",
    },
    "koris": {
        "2023-24": "https://www.oddsportal.com/basketball/finland/korisliiga-2023-2024/results/",
        "2024-25": "https://www.oddsportal.com/basketball/finland/korisliiga-2024-2025/results/",
        "2025-26": "https://www.oddsportal.com/basketball/finland/korisliiga/results/",
    },
    "nbl_cz": {
        "2023-24": "https://www.oddsportal.com/basketball/czech-republic/nbl-2023-2024/results/",
        "2024-25": "https://www.oddsportal.com/basketball/czech-republic/nbl-2024-2025/results/",
        "2025-26": "https://www.oddsportal.com/basketball/czech-republic/nbl/results/",
    },
    "hung": {
        "2023-24": "https://www.oddsportal.com/basketball/hungary/nb-i-a-2023-2024/results/",
        "2024-25": "https://www.oddsportal.com/basketball/hungary/nb-i-a-2024-2025/results/",
        "2025-26": "https://www.oddsportal.com/basketball/hungary/nb-i-a/results/",
    },
}

# Leagues most likely to have meaningful O/U coverage — scrape these first
HIGH_PRIORITY_LEAGUES = ["nba", "acb", "bbl", "bsl", "euroleague", "eurocup", "aba", "cba"]

MONTHS_EN = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Browser helpers
# ---------------------------------------------------------------------------

async def _new_stealth_page(browser):
    from playwright_stealth import Stealth
    stealth = Stealth(navigator_webdriver=True)
    context = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1280, "height": 900},
        locale="en-US",
    )
    page = await context.new_page()
    await stealth.apply_stealth_async(page)
    return page


_consent_dismissed = False


async def _dismiss_consent(page) -> None:
    global _consent_dismissed
    if _consent_dismissed:
        return
    for sel in ["button#onetrust-accept-btn-handler", ".fc-cta-consent button",
                "button[aria-label*='Accept']", "button:text('Accept All')"]:
        try:
            btn = page.locator(sel).first
            if await btn.is_visible(timeout=2000):
                await btn.click()
                await asyncio.sleep(1.5)
                _consent_dismissed = True
                return
        except Exception:
            pass


async def _switch_to_ou_market(page, delay: float = 1.5) -> bool:
    """
    Click the Over/Under market tab on a results page so that the table
    shows O/U odds instead of 1X2. Returns True if the click was found.
    """
    switched = await page.evaluate("""() => {
        const candidates = [
            ...document.querySelectorAll('a, button, div[role="tab"], span[role="tab"]')
        ];
        // Look for exact "Over/Under" text or common abbreviations
        const el = candidates.find(e => {
            const t = (e.textContent || '').trim();
            return t === 'Over/Under' || t === 'O/U' || t === 'Totals';
        });
        if (el) { el.click(); return true; }
        // Fallback: any link whose href contains 'ou'
        const ouLink = document.querySelector('a[href*="#ou"], a[href*="ou;"]');
        if (ouLink) { ouLink.click(); return true; }
        return false;
    }""")
    if switched:
        await asyncio.sleep(delay)
    return bool(switched)


async def _extract_inline_ou(page, league_slug: str) -> dict[str, float]:
    """
    After switching to the O/U market view on the results page, extract
    the O/U line for each visible game row.

    Strategy: for each game anchor (matching the league slug), walk up to
    the nearest container element and look for "Over/Under +XXX.X" text.

    Returns {game_url: ou_total}.
    """
    data: dict[str, float] = await page.evaluate(f"""() => {{
        const slug = {json.dumps(league_slug)};
        const results = {{}};

        document.querySelectorAll('a').forEach(a => {{
            const h = a.href || '';
            // Must be a game link (deep path, no hash, contains league slug)
            if (!h.includes(slug) || h.split('/').length < 7 || h.includes('#'))
                return;

            // Walk up max 6 levels to find a row container with O/U text
            let el = a;
            for (let i = 0; i < 6; i++) {{
                el = el.parentElement;
                if (!el) break;
                const text = el.innerText || '';
                // Primary pattern: "Over/Under +192.5"
                let m = text.match(/Over\\/Under \\+(\\d+\\.?\\d*)/);
                if (m) {{ results[h] = parseFloat(m[1]); return; }}
                // Fallback: standalone total near odds (e.g. "192.5\\n1.85\\n1.95")
                m = text.match(/(\\d{{2,3}}\\.5)\\s*[\\n\\r]\\s*1\\.\\d+\\s*[\\n\\r]\\s*1\\.\\d+/);
                if (m) {{ results[h] = parseFloat(m[1]); return; }}
            }}
        }});

        return results;
    }}""")
    return data or {}


# ---------------------------------------------------------------------------
# Results page — collect game URLs + try inline O/U extraction
# ---------------------------------------------------------------------------

async def get_game_urls_for_season(
    page,
    results_url: str,
    league: str,
    delay: float = 1.5,
) -> list[dict]:
    """
    Paginate through results pages and collect game metadata.

    Phase 1: On each page, switch to O/U market and extract inline odds.
    Falls back gracefully if O/U not shown in table (e.g. EuroLeague regular season).

    Returns list of dicts:
        {url, away_raw, home_raw, away_score, home_score, book_total (float or None)}
    """
    url_path = results_url.rstrip("/").rsplit("/results", 1)[0]
    league_slug = url_path.split("/")[-1]

    all_games: list[dict] = []
    seen_urls: set[str] = set()

    for page_num in range(1, 25):
        try:
            if page_num == 1:
                try:
                    await page.goto(results_url, timeout=40000, wait_until="networkidle")
                except Exception:
                    await page.goto(results_url, timeout=40000, wait_until="domcontentloaded")
                    await asyncio.sleep(delay + 2)
                await _dismiss_consent(page)
                # Switch to O/U market view for inline extraction
                switched = await _switch_to_ou_market(page, delay)
                if switched:
                    typer.echo(f"  Switched to O/U market view")
            else:
                clicked = await page.evaluate(f"""(pageNum) => {{
                    const btn = document.querySelector('a.pagination-link[data-number="' + pageNum + '"]');
                    if (!btn) return false;
                    btn.click();
                    return true;
                }}""", page_num)
                if not clicked:
                    break
                # Wait for AJAX to complete (network idle), then wait for DOM to stabilise
                try:
                    await page.wait_for_load_state("networkidle", timeout=12000)
                except Exception:
                    pass
                # Poll until at least 10 game links are visible (game URL depth > 7 segments)
                for _ in range(16):  # up to 8 s total
                    await asyncio.sleep(0.5)
                    game_link_count = await page.evaluate(f"""() => {{
                        const slug = {json.dumps(league_slug)};
                        return Array.from(document.querySelectorAll('a'))
                            .filter(a => {{
                                const h = a.href || '';
                                return h.includes(slug) && h.split('/').length > 7 && !h.includes('#');
                            }}).length;
                    }}""")
                    if game_link_count >= 10:
                        break
                await asyncio.sleep(delay * 0.5)  # final DOM settle

            await asyncio.sleep(delay * 0.3)

            # Try inline O/U extraction from the current page's table
            inline_ou = await _extract_inline_ou(page, league_slug)

            links = await page.evaluate(f"""() => {{
                const leagueSlug = {json.dumps(league_slug)};
                const seen = new Set();
                const results = [];
                document.querySelectorAll('a').forEach(a => {{
                    const h = a.href || '';
                    if (!h.includes(leagueSlug) || h.includes('#')) return;
                    // Game links end with a short alphanumeric hash segment
                    const parts = h.split('/');
                    const seg = (parts[parts.length - 1] || parts[parts.length - 2] || '');
                    // Must end with an 8-char hash (OddsPortal game slug format)
                    if (!/[A-Za-z0-9]{{6,12}}$/.test(seg)) return;
                    if (seg.includes('standings') || seg.includes('results') || seg.includes('outrights')) return;
                    if (seen.has(h)) return;
                    seen.add(h);
                    results.push({{
                        href: h,
                        text: (a.innerText||'').replace(/\\s+/g,' ').trim()
                    }});
                }});
                return results;
            }}""")

            new_games = 0
            inline_hits = 0
            for link in links:
                href = link["href"]
                if href in seen_urls:
                    continue
                parts = href.split(league_slug + "/")
                if len(parts) < 2:
                    continue
                slug_part = parts[-1].rstrip("/")
                if not slug_part or "/" in slug_part or "-" not in slug_part:
                    continue
                seen_urls.add(href)

                text = link["text"]
                parsed = _parse_game_link_text(text)
                if parsed:
                    book_total = inline_ou.get(href)
                    if book_total:
                        inline_hits += 1
                    all_games.append({
                        "url": href,
                        "away_raw": parsed["away"],
                        "home_raw": parsed["home"],
                        "away_score": parsed["away_score"],
                        "home_score": parsed["home_score"],
                        "book_total": book_total,  # None if not found inline
                    })
                    new_games += 1

            inline_note = f", {inline_hits} with inline O/U" if inline_hits else ""

            # HTML source fallback: some leagues lazy-render game links in the DOM
            # (e.g. ABA 2024-25+ after league rename). Extract paths from raw HTML instead.
            if new_games == 0:
                html = await page.content()
                base = results_url.rstrip("/").rsplit("/results", 1)[0]
                raw_paths = re.findall(
                    rf'/{re.escape(league_slug)}/([a-z0-9%-]+-[A-Za-z0-9]{{6,12}})(?:/|")',
                    html,
                )
                raw_paths = list(dict.fromkeys(raw_paths))  # deduplicate, preserve order
                for path in raw_paths:
                    game_url = f"{base}/{path}/"
                    if game_url in seen_urls:
                        continue
                    seen_urls.add(game_url)
                    all_games.append({
                        "url": game_url,
                        "away_raw": "",
                        "home_raw": "",
                        "away_score": 0,
                        "home_score": 0,
                        "book_total": None,
                    })
                    new_games += 1
                if new_games:
                    inline_note = " (HTML source fallback)"

            typer.echo(f"  Page {page_num}: {new_games} new games ({len(all_games)} total{inline_note})")

            if new_games == 0:
                break

            await asyncio.sleep(delay)

        except Exception as e:
            typer.echo(f"  Page {page_num} error: {e}", err=True)
            break

    return all_games


def _parse_game_link_text(text: str) -> dict | None:
    """
    Parse OddsPortal game link text.
    Format: "HH:MM Away Team Name NN – NN Home Team Name [OT]"
    OddsPortal lists: away first, home second.
    """
    text = re.sub(r"^\d{1,2}:\d{2}\s*", "", text).strip()
    text = re.sub(r"\s+OT\s*$", "", text).strip()

    m = re.search(r"^(.+?)\s+(\d{2,3})\s*[–\-]\s*(\d{2,3})\s+(.+)$", text)
    if not m:
        return None

    return {
        "away": m.group(1).strip(),
        "home": m.group(4).strip(),
        "away_score": int(m.group(2)),
        "home_score": int(m.group(3)),
    }


# ---------------------------------------------------------------------------
# Game page — extract O/U line and date (Phase 2 fallback)
# ---------------------------------------------------------------------------

async def get_ou_from_game_page(page, game_url: str, delay: float = 1.5) -> dict | None:
    """
    Load a game page, switch to O/U market, extract:
      - game_date
      - home_team_raw, away_team_raw (from page header)
      - book_total (consensus O/U line)

    Returns None on failure.
    """
    try:
        try:
            await page.goto(game_url, timeout=40000, wait_until="networkidle")
        except Exception:
            await page.goto(game_url, timeout=40000, wait_until="domcontentloaded")
            await asyncio.sleep(delay)
        await _dismiss_consent(page)
        await asyncio.sleep(delay * 0.4)

        # Click the Over/Under market tab — this is the reliable method
        await page.evaluate("""() => {
            const els = Array.from(document.querySelectorAll('a, div, button, span'));
            const ou = els.find(el => el.textContent.trim() === 'Over/Under');
            if (ou) ou.click();
        }""")
        await asyncio.sleep(delay)

        all_text = await page.evaluate("() => document.body.innerText")
        lines = [l.strip() for l in all_text.split("\n") if l.strip()]

        game_date = _parse_date_from_lines(lines)
        away_raw, home_raw = _parse_teams_from_lines(lines)
        ou_records = _parse_ou_lines(lines)
        book_total = _find_consensus_line(ou_records)

        if book_total is None:
            return None

        return {
            "game_date": game_date.isoformat() if game_date else "",
            "home_raw": home_raw or "",
            "away_raw": away_raw or "",
            "book_total": book_total,
            "odds_source": "oddsportal",
            "url": game_url,
        }

    except Exception as e:
        typer.echo(f"  Game page error ({game_url.split('/')[-2]}): {e}", err=True)
        return None


def _parse_date_from_lines(lines: list[str]) -> date | None:
    for line in lines:
        # Handle formats: "05 Jun 2025", "05 Jun 2025,", "5 June 2025"
        m = re.search(r"(\d{1,2})\s+([A-Za-z]+)[,.]?\s+(\d{4})", line)
        if m:
            day = int(m.group(1))
            month = MONTHS_EN.get(m.group(2).lower(), 0)
            year = int(m.group(3).rstrip(",."))
            if month and 2015 <= year <= 2030:
                try:
                    return date(year, month, day)
                except ValueError:
                    pass
    return None


def _parse_teams_from_lines(lines: list[str]) -> tuple[str, str]:
    for i, line in enumerate(lines):
        if line == "–" and i >= 2 and i + 2 < len(lines):
            away_cand = lines[i - 2]
            home_cand = lines[i + 2] if i + 2 < len(lines) else ""
            if re.match(r"^\d{2,3}$", lines[i - 1]) and re.match(r"^\d{2,3}$", lines[i + 1]):
                return away_cand, home_cand
    return "", ""


def _parse_ou_lines(lines: list[str]) -> list[dict]:
    records = []
    start = -1
    for i, line in enumerate(lines):
        if line == "Over" and i + 1 < len(lines) and lines[i + 1] == "Under":
            start = i + 3
            break
    if start < 0:
        return records

    i = start
    while i < len(lines):
        line = lines[i]
        m = re.match(r"Over/Under \+(\d+\.?\d*)", line)
        if not m:
            if line and not re.match(r"^[\d.\-]", line) and "%" not in line:
                if any(keyword in line for keyword in
                       ["1X2", "Home/Away", "Asian", "Double", "Odd or Even",
                        "Quarter", "Half", "All Bonuses", "Bookmakers", "More"]):
                    break
            i += 1
            continue

        total = float(m.group(1))
        i += 1

        books = 1
        over_odds = under_odds = 0.0

        if i < len(lines) and re.match(r"^\d{1,2}$", lines[i]):
            books = int(lines[i])
            i += 1

        for _ in range(3):
            if i >= len(lines):
                break
            val = lines[i]
            if re.match(r"^\d+\.\d+$", val):
                if over_odds == 0.0:
                    over_odds = float(val)
                elif under_odds == 0.0:
                    under_odds = float(val)
            elif val == "-":
                pass
            elif "%" in val:
                i += 1
                break
            i += 1

        if over_odds > 0 and under_odds > 0:
            records.append({
                "total": total,
                "books": books,
                "over": over_odds,
                "under": under_odds,
            })

    return records


def _find_consensus_line(records: list[dict]) -> float | None:
    if not records:
        return None
    best = sorted(records, key=lambda r: (-r["books"], abs(r["over"] - r["under"])))
    return best[0]["total"]


# ---------------------------------------------------------------------------
# Main scraping loop
# ---------------------------------------------------------------------------

async def scrape_league_season(
    browser,
    league: str,
    season: str,
    results_url: str,
    bronze_dir: Path,
    delay: float,
    skip_game_pages: bool = False,
    max_no_odds_streak: int = 20,
) -> list[dict]:
    """
    Scrape one league/season. Results cached to bronze JSON.

    Phase 1: Collect game stubs from results page (with inline O/U if available).
    Phase 2: For games without inline O/U, visit individual game pages.
             Skipped when skip_game_pages=True.
             Stops early if max_no_odds_streak consecutive games have no O/U
             (indicates bookmakers don't cover this league — bail out fast).
    """
    cache_path = bronze_dir / league / f"odds_{season.replace('/', '-')}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached: dict[str, dict] = {}
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text())
            cached = {r["url"]: r for r in raw if "url" in r}
            typer.echo(f"  Loaded {len(cached)} cached games from {cache_path.name}")
        except Exception:
            pass

    page = await _new_stealth_page(browser)
    global _consent_dismissed
    _consent_dismissed = False

    try:
        typer.echo(f"\n[{league.upper()} {season}] Collecting game URLs...")
        game_stubs = await get_game_urls_for_season(page, results_url, league, delay=delay)
        typer.echo(f"  Found {len(game_stubs)} games total")

        results: list[dict] = list(cached.values())

        # ── Phase 1: Use inline O/U from results table ────────────────────
        inline_count = 0
        needs_game_page: list[dict] = []

        for stub in game_stubs:
            game_url = stub["url"]
            if game_url in cached:
                continue

            if stub.get("book_total") is not None:
                # We have inline O/U — build a record without visiting the game page.
                # Date comes from the link text score (we know teams + approximate season).
                # We'll fill game_date as empty string; patch_gold_with_alt_ids matches by slug.
                rec = {
                    "game_date": "",           # filled below if we can parse from context
                    "home_raw": stub["home_raw"],
                    "away_raw": stub["away_raw"],
                    "book_total": stub["book_total"],
                    "odds_source": "oddsportal",
                    "url": game_url,
                    "league": league,
                    "season": season,
                }
                results.append(rec)
                cached[game_url] = rec
                inline_count += 1
            else:
                needs_game_page.append(stub)

        if inline_count:
            typer.echo(f"  Phase 1: {inline_count} games with inline O/U (no game-page visits needed)")
            cache_path.write_text(json.dumps(list(cached.values()), indent=2))

        # ── Phase 2: Per-game pages for remaining games ───────────────────
        if skip_game_pages:
            typer.echo(f"  Phase 2 skipped (--no-game-pages)")
        elif needs_game_page:
            typer.echo(f"  Phase 2: fetching {len(needs_game_page)} game pages for O/U...")
            new_count = 0
            no_odds_streak = 0

            for stub in needs_game_page:
                game_url = stub["url"]
                if game_url in cached:
                    continue

                # Bail out early if too many consecutive games have no O/U
                if no_odds_streak >= max_no_odds_streak:
                    typer.echo(
                        f"  {no_odds_streak} consecutive games with no O/U — "
                        f"bookmakers likely don't cover {league.upper()}. Stopping early."
                    )
                    break

                typer.echo(f"  Fetching O/U: {stub.get('away_raw', '?')} @ {stub.get('home_raw', '?')}",
                           nl=False)
                rec = await get_ou_from_game_page(page, game_url, delay=delay)

                if rec:
                    if not rec["away_raw"] and stub.get("away_raw"):
                        rec["away_raw"] = stub["away_raw"]
                    if not rec["home_raw"] and stub.get("home_raw"):
                        rec["home_raw"] = stub["home_raw"]
                    rec["league"] = league
                    rec["season"] = season
                    results.append(rec)
                    cached[game_url] = rec
                    new_count += 1
                    no_odds_streak = 0
                    typer.echo(f"  → {rec['book_total']}")
                else:
                    no_odds_streak += 1
                    typer.echo("  → no O/U")

                if new_count % 10 == 0 and new_count > 0:
                    cache_path.write_text(json.dumps(list(cached.values()), indent=2))

                await asyncio.sleep(delay * 0.3)

        cache_path.write_text(json.dumps(list(cached.values()), indent=2))
        typer.echo(f"\n  Saved {len(cached)} records → {cache_path}")
        return results

    finally:
        await page.context.close()


# ---------------------------------------------------------------------------
# Silver odds parquet builder
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def build_silver_odds(bronze_dir: Path) -> int:
    """
    Read all bronze odds JSON files → build/update silver/odds.parquet.
    Returns number of records written.
    """
    import pandas as pd

    all_records = []
    for json_path in sorted(bronze_dir.rglob("odds_*.json")):
        try:
            rows = json.loads(json_path.read_text())
            all_records.extend(rows)
        except Exception as e:
            typer.echo(f"  Skipping {json_path.name}: {e}", err=True)

    if not all_records:
        typer.echo("No bronze odds records found.")
        return 0

    rows_out = []
    for rec in all_records:
        try:
            raw_date = rec.get("game_date", "")
            game_date = date.fromisoformat(raw_date) if raw_date else None
            home_raw = rec.get("home_raw", "")
            away_raw = rec.get("away_raw", "")
            book_total = float(rec["book_total"])

            if not home_raw or not away_raw or book_total <= 0:
                continue

            rows_out.append({
                "date": game_date.isoformat() if game_date else "",
                "home_raw": home_raw,
                "away_raw": away_raw,
                "home_slug": _slugify(home_raw),
                "away_slug": _slugify(away_raw),
                "book_total": book_total,
                "odds_source": rec.get("odds_source", "oddsportal"),
                "league": rec.get("league", ""),
                "season": rec.get("season", ""),
            })
        except Exception:
            continue

    if not rows_out:
        typer.echo("No valid records to write.")
        return 0

    df = pd.DataFrame(rows_out)
    # Drop duplicates; for inline records without date, dedupe by teams + total
    df = df.drop_duplicates(subset=["date", "home_slug", "away_slug"])

    silver_dir = ROOT / "data" / "silver"
    silver_dir.mkdir(parents=True, exist_ok=True)
    odds_path = silver_dir / "odds.parquet"

    if odds_path.exists():
        existing = pd.read_parquet(odds_path)
        if "odds_source" in existing.columns:
            keep = existing[existing["odds_source"] != "oddsportal"]
            df = pd.concat([keep, df], ignore_index=True)

    df.to_parquet(odds_path, index=False, engine="pyarrow")
    typer.echo(f"Silver odds saved: {len(df)} records → {odds_path}")
    return len(df)


# ---------------------------------------------------------------------------
# Gold merge helper
# ---------------------------------------------------------------------------

def patch_gold_with_alt_ids(silver_dir: Path, gold_path: Path) -> None:
    """
    Match OddsPortal odds to gold table using date + fuzzy team name matching.
    For inline-extracted records with no date, matches by team slugs only.
    """
    import pandas as pd

    odds_path = silver_dir / "odds.parquet"
    if not odds_path.exists() or not gold_path.exists():
        return

    odds_df = pd.read_parquet(odds_path)
    gold_df = pd.read_parquet(gold_path)

    op = odds_df[odds_df.get("odds_source", pd.Series([""] * len(odds_df))) == "oddsportal"].copy() \
        if "odds_source" in odds_df.columns else odds_df.copy()

    if op.empty or "home_slug" not in op.columns:
        typer.echo("No OddsPortal records with slug columns found.")
        return

    if "book_total" not in gold_df.columns:
        gold_df["book_total"] = float("nan")
    if "home_team_id" not in gold_df.columns or "away_team_id" not in gold_df.columns:
        typer.echo("Gold table missing team_id columns — skipping fuzzy match.")
        return

    gold_df["_date_str"] = pd.to_datetime(gold_df["date"]).dt.strftime("%Y-%m-%d")
    gold_by_date = gold_df.groupby("_date_str")

    def _slug_matches(short_slug: str, full_id: str) -> bool:
        return full_id.startswith(short_slug) or short_slug in full_id

    matched_count = 0
    for _, row in op.iterrows():
        raw_date = str(row.get("date", ""))[:10]
        h_slug = str(row["home_slug"])
        a_slug = str(row["away_slug"])
        book_total = float(row["book_total"])

        if raw_date and raw_date in gold_by_date.groups:
            # Prefer date-exact matches
            candidates = gold_df.loc[gold_by_date.groups[raw_date]]
        else:
            # No date: search all rows with matching league/season
            league = str(row.get("league", ""))
            season = str(row.get("season", ""))
            mask = pd.Series([True] * len(gold_df), index=gold_df.index)
            if league and "league" in gold_df.columns:
                mask &= gold_df["league"] == league
            candidates = gold_df[mask]

        for idx, g in candidates.iterrows():
            if pd.notna(g["book_total"]):
                continue
            h_id = str(g.get("home_team_id", ""))
            a_id = str(g.get("away_team_id", ""))
            if _slug_matches(h_slug, h_id) and _slug_matches(a_slug, a_id):
                gold_df.at[idx, "book_total"] = book_total
                matched_count += 1
                break
            if _slug_matches(h_slug, a_id) and _slug_matches(a_slug, h_id):
                gold_df.at[idx, "book_total"] = book_total
                matched_count += 1
                break

    gold_df.drop(columns=["_date_str"], inplace=True)
    gold_df.to_parquet(gold_path, index=False, engine="pyarrow")
    total_with_odds = gold_df["book_total"].notna().sum()
    typer.echo(
        f"Gold patched: {matched_count} new matches, "
        f"{total_with_odds}/{len(gold_df)} games have book_total"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False)


@app.command()
def main(
    league: Optional[str] = typer.Option(None, "--league", help=f"League to scrape (default: all). Valid: {sorted(LEAGUE_SEASON_URLS)}"),
    season: Optional[str] = typer.Option(None, "--season", help="Season e.g. 2023-24 (default: all)"),
    delay: float = typer.Option(1.8, "--delay", help="Seconds between requests"),
    no_game_pages: bool = typer.Option(False, "--no-game-pages", help="Phase 1 only: use inline O/U from results table, skip per-game page visits (fast)"),
    max_no_odds_streak: int = typer.Option(20, "--max-no-odds-streak", help="Stop visiting game pages after N consecutive games with no O/U"),
    build_silver_only: bool = typer.Option(False, "--build-silver-only", help="Skip scraping, just rebuild parquet from existing bronze JSON"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print what would be scraped without doing it"),
    high_priority_only: bool = typer.Option(False, "--high-priority", help=f"Only scrape high-priority leagues: {HIGH_PRIORITY_LEAGUES}"),
) -> None:
    """Scrape OddsPortal historical O/U lines → silver/odds.parquet."""
    bronze_dir = ROOT / "data" / "bronze" / "odds" / "history"
    gold_path = ROOT / "data" / "gold" / "features.parquet"
    silver_dir = ROOT / "data" / "silver"

    if build_silver_only:
        typer.echo("Building silver odds parquet from existing bronze JSON...")
        n = build_silver_odds(bronze_dir)
        if n > 0:
            patch_gold_with_alt_ids(silver_dir, gold_path)
        return

    targets: list[tuple[str, str, str]] = []
    if league:
        leagues = [league]
    elif high_priority_only:
        leagues = [lg for lg in HIGH_PRIORITY_LEAGUES if lg in LEAGUE_SEASON_URLS]
    else:
        # Default order: high-priority first, then the rest
        others = [lg for lg in LEAGUE_SEASON_URLS if lg not in HIGH_PRIORITY_LEAGUES]
        leagues = HIGH_PRIORITY_LEAGUES + others

    for lg in leagues:
        if lg not in LEAGUE_SEASON_URLS:
            typer.echo(f"Unknown league: {lg!r}. Valid: {sorted(LEAGUE_SEASON_URLS)}", err=True)
            raise typer.Exit(1)
        seasons = [season] if season else list(LEAGUE_SEASON_URLS[lg].keys())
        for s in seasons:
            if s not in LEAGUE_SEASON_URLS[lg]:
                typer.echo(f"Unknown season {s!r} for {lg}. Valid: {list(LEAGUE_SEASON_URLS[lg].keys())}", err=True)
                raise typer.Exit(1)
            targets.append((lg, s, LEAGUE_SEASON_URLS[lg][s]))

    typer.echo(f"Targets: {len(targets)} league/season combos")
    for lg, s, url in targets:
        typer.echo(f"  {lg} {s} → {url}")

    if dry_run:
        return

    asyncio.run(_run_scraping(targets, bronze_dir, silver_dir, gold_path, delay, no_game_pages, max_no_odds_streak))


async def _run_scraping(
    targets: list[tuple[str, str, str]],
    bronze_dir: Path,
    silver_dir: Path,
    gold_path: Path,
    delay: float,
    skip_game_pages: bool,
    max_no_odds_streak: int,
) -> None:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            for league, season, results_url in targets:
                await scrape_league_season(
                    browser, league, season, results_url, bronze_dir, delay,
                    skip_game_pages=skip_game_pages,
                    max_no_odds_streak=max_no_odds_streak,
                )
        finally:
            await browser.close()

    typer.echo("\n── Building silver odds parquet... ──")
    n = build_silver_odds(bronze_dir)
    if n > 0:
        patch_gold_with_alt_ids(silver_dir, gold_path)
        typer.echo("\nRun `python cli.py build-gold && python cli.py train` to retrain with odds data.")


if __name__ == "__main__":
    app()
