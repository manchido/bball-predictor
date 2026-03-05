"""
Scrape historical over/under closing lines from OddsPortal.

Covers all 5 leagues × 3 seasons. Results are cached to bronze JSON
and then written to data/silver/odds.parquet for automatic inclusion
in the gold feature table (book_total column).

Usage:
    python scripts/scrape_odds_oddsportal.py                          # all leagues/seasons
    python scripts/scrape_odds_oddsportal.py --league euroleague      # one league, all seasons
    python scripts/scrape_odds_oddsportal.py --league euroleague --season 2023-24
    python scripts/scrape_odds_oddsportal.py --build-silver-only      # skip scraping, rebuild parquet
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
LEAGUE_SEASON_URLS: dict[str, dict[str, str]] = {
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
}

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


# ---------------------------------------------------------------------------
# Results page — get all game URLs
# ---------------------------------------------------------------------------

async def get_game_urls_for_season(
    page,
    results_url: str,
    league: str,
    delay: float = 1.5,
) -> list[dict]:
    """
    Paginate through results pages and collect game URLs + team/score info.
    Returns list of dicts: {url, away_raw, home_raw, away_score, home_score}.
    """
    # Determine the URL fragment that identifies game links
    # OddsPortal game URLs: .../league-YYYY-YYYY/{slug}-{hash}/
    # We match links that are deeper than the results page itself.
    base_domain = "oddsportal.com"
    all_games: list[dict] = []
    seen_urls: set[str] = set()

    # Parse the base path to build the correct filter
    # e.g. results_url = "https://www.oddsportal.com/basketball/europe/euroleague-2023-2024/results/"
    # league slug in links will contain "euroleague-2023-2024/" or just "euroleague/" for current
    url_path = results_url.rstrip("/").rsplit("/results", 1)[0]  # ".../euroleague-2023-2024"
    league_slug = url_path.split("/")[-1]  # "euroleague-2023-2024"

    for page_num in range(1, 25):
        try:
            if page_num == 1:
                # Initial navigation to results page
                try:
                    await page.goto(results_url, timeout=40000, wait_until="networkidle")
                except Exception:
                    await page.goto(results_url, timeout=40000, wait_until="domcontentloaded")
                    await asyncio.sleep(delay + 2)
                await _dismiss_consent(page)
            else:
                # Click the pagination button for this page number
                clicked = await page.evaluate(f"""(pageNum) => {{
                    const btns = Array.from(document.querySelectorAll('a.pagination-link'));
                    const btn = btns.find(b => b.innerText.trim() === String(pageNum));
                    if (!btn) return false;
                    btn.click();
                    return true;
                }}""", page_num)
                if not clicked:
                    break  # No more pages
                # Wait for AJAX — retry up to 3 times if page appears empty
                for _attempt in range(3):
                    await asyncio.sleep(delay + 2)
                    test_links = await page.evaluate(f"""() => {{
                        const slug = {json.dumps(league_slug)};
                        return Array.from(document.querySelectorAll('a'))
                            .filter(a => a.href.includes(slug) && a.href.split('/').length >= 7 && !a.href.includes('#'))
                            .length;
                    }}""")
                    if test_links > 0:
                        break
                    # Page might be transitioning, wait more
                    await asyncio.sleep(2)
            await asyncio.sleep(delay * 0.3)

            links = await page.evaluate(f"""() => {{
                const leagueSlug = {json.dumps(league_slug)};
                const all = Array.from(document.querySelectorAll('a'));
                return all
                    .filter(a => {{
                        const h = a.href || '';
                        return h.includes(leagueSlug) &&
                               h.split('/').length >= 7 &&
                               !h.endsWith('/results/') &&
                               !h.endsWith('/standings/') &&
                               !h.endsWith('/outrights/') &&
                               !h.includes('#');
                    }})
                    .map(a => ({{
                        href: a.href,
                        text: (a.innerText||'').replace(/\\s+/g,' ').trim()
                    }}));
            }}""")

            # Filter: keep only individual game links (one slug segment after the league path)
            new_games = 0
            for link in links:
                href = link["href"]
                if href in seen_urls:
                    continue
                # Split on the league slug to get the game slug
                parts = href.split(league_slug + "/")
                if len(parts) < 2:
                    continue
                slug_part = parts[-1].rstrip("/")
                # A game slug looks like "team1-team2-HASH8" — no extra slashes, not empty
                if not slug_part or "/" in slug_part:
                    continue
                # Must contain a hyphen (team names) — filters out bare league root links
                if "-" not in slug_part:
                    continue
                seen_urls.add(href)

                # Parse team names and score from link text
                # Format: "HH:MM Away Name NN – NN Home Name [OT]"
                text = link["text"]
                parsed = _parse_game_link_text(text)
                if parsed:
                    all_games.append({
                        "url": href,
                        "away_raw": parsed["away"],
                        "home_raw": parsed["home"],
                        "away_score": parsed["away_score"],
                        "home_score": parsed["home_score"],
                    })
                    new_games += 1

            typer.echo(f"  Page {page_num}: {new_games} new games ({len(all_games)} total)")

            if new_games == 0:
                # No new games — end of results
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
    # Strip time prefix: "15:30 Zalgiris Kaunas 87 – 73 Olimpia Milano"
    text = re.sub(r"^\d{1,2}:\d{2}\s*", "", text).strip()
    text = re.sub(r"\s+OT\s*$", "", text).strip()

    # Try to split on score pattern " NN – NN " or " NN - NN "
    m = re.search(r"^(.+?)\s+(\d{2,3})\s*[–\-]\s*(\d{2,3})\s+(.+)$", text)
    if not m:
        return None

    away_name = m.group(1).strip()
    away_score = int(m.group(2))
    home_score = int(m.group(3))
    home_name = m.group(4).strip()

    return {
        "away": away_name,
        "home": home_name,
        "away_score": away_score,
        "home_score": home_score,
    }


# ---------------------------------------------------------------------------
# Game page — extract O/U line and date
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

        # Trigger the O/U market view via JS hash
        await page.evaluate("() => { window.location.hash = '#ou;2'; }")
        await asyncio.sleep(delay)

        # Also try clicking Over/Under via JS
        await page.evaluate("""() => {
            const els = Array.from(document.querySelectorAll('a, div, button, span'));
            const ou = els.find(el => el.textContent.trim() === 'Over/Under');
            if (ou) ou.click();
        }""")
        await asyncio.sleep(delay * 0.5)

        all_text = await page.evaluate("() => document.body.innerText")
        lines = [l.strip() for l in all_text.split("\n") if l.strip()]

        # --- Extract game date ---
        game_date = _parse_date_from_lines(lines)

        # --- Extract team names from page header ---
        away_raw, home_raw = _parse_teams_from_lines(lines)

        # --- Extract O/U lines ---
        ou_records = _parse_ou_lines(lines)
        book_total = _find_consensus_line(ou_records)

        if book_total is None or game_date is None:
            return None

        return {
            "game_date": game_date.isoformat(),
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
    """Parse game date from page text lines."""
    for line in lines:
        # Pattern: "26 May 2024," or "26 May 2024"
        m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", line)
        if m:
            day = int(m.group(1))
            month = MONTHS_EN.get(m.group(2).lower(), 0)
            year = int(m.group(3))
            if month and 2015 <= year <= 2030:
                try:
                    return date(year, month, day)
                except ValueError:
                    pass
    return None


def _parse_teams_from_lines(lines: list[str]) -> tuple[str, str]:
    """
    Extract away and home team names from game page lines.
    OddsPortal header: "Away Team\nSCORE\n–\nSCORE\nHome Team"
    Returns (away_raw, home_raw).
    """
    for i, line in enumerate(lines):
        if line == "–" and i >= 2 and i + 2 < len(lines):
            # Lines: [away, away_score, "–", home_score, home]
            away_cand = lines[i - 2]
            home_cand = lines[i + 2] if i + 2 < len(lines) else ""
            # Validate: adjacent lines should be numeric scores
            if re.match(r"^\d{2,3}$", lines[i - 1]) and re.match(r"^\d{2,3}$", lines[i + 1]):
                return away_cand, home_cand
    return "", ""


def _parse_ou_lines(lines: list[str]) -> list[dict]:
    """
    Parse Over/Under lines from page text.
    Structure after market switch:
      Over  Under  Payout
      Over/Under +X.X  books  over_odds  under_odds  payout%
    """
    records = []

    # Find start of O/U section
    start = -1
    for i, line in enumerate(lines):
        if line == "Over" and i + 1 < len(lines) and lines[i + 1] == "Under":
            start = i + 3  # skip "Over", "Under", "Payout"
            break
    if start < 0:
        return records

    i = start
    while i < len(lines):
        line = lines[i]
        m = re.match(r"Over/Under \+(\d+\.?\d*)", line)
        if not m:
            # Stop if we hit something clearly outside the OU table
            if line and not re.match(r"^[\d.\-]", line) and "%" not in line:
                if any(keyword in line for keyword in
                       ["1X2", "Home/Away", "Asian", "Double", "Odd or Even",
                        "Quarter", "Half", "All Bonuses", "Bookmakers", "More"]):
                    break
            i += 1
            continue

        total = float(m.group(1))
        i += 1

        # Read: books_count, over_odds, under_odds, payout%
        books = 1
        over_odds = under_odds = 0.0

        if i < len(lines) and re.match(r"^\d{1,2}$", lines[i]):
            books = int(lines[i])
            i += 1

        for _ in range(3):  # up to 3 values: over, under, payout
            if i >= len(lines):
                break
            val = lines[i]
            if re.match(r"^\d+\.\d+$", val):
                if over_odds == 0.0:
                    over_odds = float(val)
                elif under_odds == 0.0:
                    under_odds = float(val)
            elif val == "-":
                pass  # no odds for this line
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
    """
    Pick the consensus O/U line from a list of bookmaker lines.

    Strategy:
      1. Prefer lines with the most bookmakers offering odds
      2. Among those, prefer the most balanced line (|over - under| closest to 0)
    """
    if not records:
        return None

    # Sort by: most books first, then most balanced (smallest |over-under|)
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
) -> list[dict]:
    """Scrape one league/season. Results cached to bronze JSON."""
    cache_path = bronze_dir / league / f"odds_{season.replace('/', '-')}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache
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
        # Step 1: collect all game URLs from results pages
        typer.echo(f"\n[{league.upper()} {season}] Collecting game URLs...")
        game_stubs = await get_game_urls_for_season(page, results_url, league, delay=delay)
        typer.echo(f"  Found {len(game_stubs)} games total")

        # Step 2: for each game, fetch O/U line if not cached
        results: list[dict] = list(cached.values())
        new_count = 0

        for stub in game_stubs:
            game_url = stub["url"]
            if game_url in cached:
                continue  # already scraped

            typer.echo(f"  Fetching O/U: {stub.get('away_raw', '?')} @ {stub.get('home_raw', '?')}",
                       nl=False)
            rec = await get_ou_from_game_page(page, game_url, delay=delay)

            if rec:
                # Prefer teams from the results page (already parsed) if game page parsing fails
                if not rec["away_raw"] and stub.get("away_raw"):
                    rec["away_raw"] = stub["away_raw"]
                if not rec["home_raw"] and stub.get("home_raw"):
                    rec["home_raw"] = stub["home_raw"]
                rec["league"] = league
                rec["season"] = season
                results.append(rec)
                cached[game_url] = rec
                new_count += 1
                typer.echo(f"  → {rec['book_total']}")
            else:
                typer.echo("  → FAILED")

            # Save cache periodically
            if new_count % 10 == 0:
                cache_path.write_text(json.dumps(list(cached.values()), indent=2))

            await asyncio.sleep(delay * 0.3)  # short gap between games on same page

        # Final save
        cache_path.write_text(json.dumps(list(cached.values()), indent=2))
        typer.echo(f"\n  Saved {len(cached)} records → {cache_path}")
        return results

    finally:
        await page.context.close()


# ---------------------------------------------------------------------------
# Silver odds parquet builder
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    """Normalize a team name to a slug for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def build_silver_odds(bronze_dir: Path) -> int:
    """
    Read all bronze odds JSON files → build/update silver/odds.parquet.
    Stores raw team names + slugs so patch_gold_with_alt_ids can do fuzzy matching.
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
            game_date = date.fromisoformat(rec["game_date"])
            home_raw = rec.get("home_raw", "")
            away_raw = rec.get("away_raw", "")
            book_total = float(rec["book_total"])

            if not home_raw or not away_raw or book_total <= 0:
                continue

            rows_out.append({
                "date": game_date.isoformat(),
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

    df = pd.DataFrame(rows_out).drop_duplicates(subset=["date", "home_slug", "away_slug"])

    silver_dir = ROOT / "data" / "silver"
    silver_dir.mkdir(parents=True, exist_ok=True)
    odds_path = silver_dir / "odds.parquet"

    # Merge with existing silver odds if present (keep non-oddsportal rows that have game_id for API odds)
    if odds_path.exists():
        existing = pd.read_parquet(odds_path)
        if "odds_source" in existing.columns:
            # Keep API-based odds (game_id present), replace oddsportal ones
            keep = existing[existing["odds_source"] != "oddsportal"]
            df = pd.concat([keep, df], ignore_index=True)

    df.to_parquet(odds_path, index=False, engine="pyarrow")
    typer.echo(f"Silver odds saved: {len(df)} records → {odds_path}")
    return len(df)


# ---------------------------------------------------------------------------
# Gold merge helper — also tries game_id_alt for matching
# ---------------------------------------------------------------------------

def patch_gold_with_alt_ids(silver_dir: Path, gold_path: Path) -> None:
    """
    Match OddsPortal odds to gold table using date + fuzzy team name matching.
    OddsPortal uses short names (e.g. 'Fenerbahce') while gold uses full API names
    (e.g. 'fenerbahce-beko-istanbul'). We match by checking if the short slug is a
    prefix/substring of the full team_id slug.
    """
    import pandas as pd

    odds_path = silver_dir / "odds.parquet"
    if not odds_path.exists() or not gold_path.exists():
        return

    odds_df = pd.read_parquet(odds_path)
    gold_df = pd.read_parquet(gold_path)

    # Only process OddsPortal records with slug columns
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

    # Index gold by date string for fast lookup
    gold_df["_date_str"] = pd.to_datetime(gold_df["date"]).dt.strftime("%Y-%m-%d")
    gold_by_date = gold_df.groupby("_date_str")

    def _slug_matches(short_slug: str, full_id: str) -> bool:
        """True if short slug is a prefix or substring of the full team_id."""
        return full_id.startswith(short_slug) or short_slug in full_id

    matched_count = 0
    for _, row in op.iterrows():
        date_str = str(row["date"])[:10]
        if date_str not in gold_by_date.groups:
            continue
        candidates = gold_df.loc[gold_by_date.groups[date_str]]
        h_slug = str(row["home_slug"])
        a_slug = str(row["away_slug"])
        book_total = float(row["book_total"])

        for idx, g in candidates.iterrows():
            if pd.notna(g["book_total"]):
                continue  # already filled
            h_id = str(g.get("home_team_id", ""))
            a_id = str(g.get("away_team_id", ""))
            # Normal orientation
            if _slug_matches(h_slug, h_id) and _slug_matches(a_slug, a_id):
                gold_df.at[idx, "book_total"] = book_total
                matched_count += 1
                break
            # Swapped orientation (OddsPortal sometimes swaps home/away for neutral sites)
            if _slug_matches(h_slug, a_id) and _slug_matches(a_slug, h_id):
                gold_df.at[idx, "book_total"] = book_total
                matched_count += 1
                break

    gold_df.drop(columns=["_date_str"], inplace=True)
    gold_df.to_parquet(gold_path, index=False, engine="pyarrow")
    total_with_odds = gold_df["book_total"].notna().sum()
    typer.echo(f"Gold patched: {matched_count} new matches, {total_with_odds}/{len(gold_df)} games have book_total")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False)


@app.command()
def main(
    league: Optional[str] = typer.Option(None, "--league", help="League to scrape (default: all)"),
    season: Optional[str] = typer.Option(None, "--season", help="Season e.g. 2023-24 (default: all)"),
    delay: float = typer.Option(1.8, "--delay", help="Seconds between requests"),
    build_silver_only: bool = typer.Option(False, "--build-silver-only", help="Skip scraping, just rebuild parquet"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print what would be scraped without doing it"),
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

    # Determine what to scrape
    targets: list[tuple[str, str, str]] = []
    leagues = [league] if league else list(LEAGUE_SEASON_URLS.keys())
    for lg in leagues:
        if lg not in LEAGUE_SEASON_URLS:
            typer.echo(f"Unknown league: {lg}. Valid: {list(LEAGUE_SEASON_URLS.keys())}", err=True)
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

    asyncio.run(_run_scraping(targets, bronze_dir, silver_dir, gold_path, delay))


async def _run_scraping(
    targets: list[tuple[str, str, str]],
    bronze_dir: Path,
    silver_dir: Path,
    gold_path: Path,
    delay: float,
) -> None:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            for league, season, results_url in targets:
                await scrape_league_season(
                    browser, league, season, results_url, bronze_dir, delay
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
