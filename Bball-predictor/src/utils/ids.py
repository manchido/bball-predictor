"""
Stable ID generation and team-name normalisation.

Handles the large number of name variants that appear across RealGM,
The Odds API, and national league sources for the same club.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date
from typing import Optional

from rapidfuzz import process, fuzz

# ---------------------------------------------------------------------------
# Canonical team registry
# Each entry: canonical_name → team_id (slug)
# Aliases are resolved via fuzzy matching fallback.
# ---------------------------------------------------------------------------
_CANONICAL: dict[str, str] = {
    # EuroLeague
    "Real Madrid": "real-madrid",
    "FC Barcelona": "barcelona",
    "Anadolu Efes": "efes",
    "Fenerbahce Beko": "fenerbahce",
    "Olympiacos": "olympiacos",
    "Panathinaikos": "panathinaikos",
    "CSKA Moscow": "cska-moscow",
    "Maccabi Tel Aviv": "maccabi-tel-aviv",
    "Bayern Munich": "bayern-munich",
    "Alba Berlin": "alba-berlin",
    "Baskonia": "baskonia",
    "Valencia Basket": "valencia",
    "Zalgiris Kaunas": "zalgiris",
    "Virtus Bologna": "virtus-bologna",
    "Monaco": "monaco",
    "Partizan": "partizan",
    "Milan": "milan",
    "Red Star Belgrade": "red-star",
    "Paris Basketball": "paris",
    "Asvel": "asvel",

    # EuroCup
    "Gran Canaria": "gran-canaria",
    "Joventut": "joventut",
    "Bursaspor": "bursaspor",
    "Hapoel Tel Aviv": "hapoel-tel-aviv",
    "Trento": "trento",
    "Cedevita Olimpija": "olimpija",
    "Promitheas": "promitheas",
    "Turk Telekom": "turk-telekom",
    "Tofas": "tofas",
    "Bahcesehir": "bahcesehir",
    "Galatasaray": "galatasaray",
    "Hamburg Towers": "hamburg",
    "Ratiopharm Ulm": "ulm",
    "Manresa": "manresa",
    "Dolomiti Energia": "trento",

    # ACB (Spain)
    "Unicaja": "unicaja",
    "Bilbao Basket": "bilbao",
    "Casademont Zaragoza": "zaragoza",
    "Basquet Girona": "girona",
    "Rio Breogan": "breogan",

    # BSL (Turkey)
    "Pinar Karsiyaka": "karsiyaka",
    "Beşiktaş": "besiktas",
    "Darussafaka": "darussafaka",

    # BBL (Germany)
    "Berlin": "berlin",
    "Bonn": "bonn",
    "Ludwigsburg": "ludwigsburg",
    "Braunschweig": "braunschweig",
    "Frankfurt": "frankfurt",
    "MHP Riesen Ludwigsburg": "ludwigsburg",
    "EWE Baskets Oldenburg": "oldenburg",
    "Telekom Baskets Bonn": "bonn",
    "ratiopharm ulm": "ulm",
}

# Alias patches — exact overrides before fuzzy matching
_ALIASES: dict[str, str] = {
    "real madrid basketball": "real-madrid",
    "fc barcelona basketball": "barcelona",
    "barca basket": "barcelona",
    "fc barcelona basquet": "barcelona",
    "ef istanbul": "efes",
    "anadolu efes istanbul": "efes",
    "fenerbahce basketball": "fenerbahce",
    "fenerbahce ulker": "fenerbahce",
    "olympiacos piraeus": "olympiacos",
    "panathinaikos aktor": "panathinaikos",
    "armani milan": "milan",
    "ax armani exchange milan": "milan",
    "ea7 emporio armani milan": "milan",
    "cska": "cska-moscow",
    "maccabi playtika tel aviv": "maccabi-tel-aviv",
    "fc bayern munich": "bayern-munich",
    "fc bayern": "bayern-munich",
    "ldlc asvel": "asvel",
    "ldlc asvel villeurbanne": "asvel",
    "monaco basket": "monaco",
    "as monaco": "monaco",
    "partizan mozzart bet belgrade": "partizan",
    "crvena zvezda": "red-star",
    "red star mts": "red-star",
}

_CANONICAL_NAMES = list(_CANONICAL.keys())


def normalize_team_name(raw: str) -> str:
    """Return the canonical team name for a raw string."""
    cleaned = re.sub(r"\s+", " ", raw.strip())
    lower = cleaned.lower()

    # 1. Exact alias match
    if lower in _ALIASES:
        team_id = _ALIASES[lower]
        # Reverse-lookup canonical name from id
        for name, tid in _CANONICAL.items():
            if tid == team_id:
                return name
        return cleaned  # fallback

    # 2. Exact canonical match (case-insensitive)
    for name in _CANONICAL_NAMES:
        if name.lower() == lower:
            return name

    # 3. Fuzzy match (threshold 80)
    result = process.extractOne(cleaned, _CANONICAL_NAMES, scorer=fuzz.token_sort_ratio)
    if result and result[1] >= 80:
        return result[0]

    # 4. No match — return cleaned original (will get a hashed id)
    return cleaned


def get_team_id(raw: str) -> str:
    """Return stable team_id slug for a raw team name."""
    lower = re.sub(r"\s+", " ", raw.strip()).lower()

    # Alias lookup
    if lower in _ALIASES:
        return _ALIASES[lower]

    canonical = normalize_team_name(raw)
    if canonical in _CANONICAL:
        return _CANONICAL[canonical]

    # Unknown team — generate deterministic slug from name
    slug = re.sub(r"[^a-z0-9]+", "-", canonical.lower()).strip("-")
    return slug


def make_game_id(game_date: date, home_raw: str, away_raw: str) -> str:
    """Return a stable game_id string."""
    home_id = get_team_id(home_raw)
    away_id = get_team_id(away_raw)
    date_str = game_date.isoformat()
    raw = f"{date_str}__{home_id}__{away_id}"
    # Prefix with short hash to handle edge-case double-headers
    digest = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{date_str}__{home_id}__{away_id}__{digest}"


def register_team(canonical_name: str, team_id: str, aliases: Optional[list[str]] = None) -> None:
    """Runtime registration for leagues/teams not in the static registry."""
    _CANONICAL[canonical_name] = team_id
    _CANONICAL_NAMES.append(canonical_name)
    if aliases:
        for alias in aliases:
            _ALIASES[alias.lower()] = team_id
