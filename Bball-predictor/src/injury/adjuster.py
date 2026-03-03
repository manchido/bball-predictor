"""
Injury Adjustment Module.

Identifies a team's top-3 players by season-to-date usage% × avg_minutes
and applies a PPP-proxy reduction if any of them are unavailable.

Assumptions (documented):
  - Each top-3 player contributes roughly 3% of team PPP when playing.
  - Injury proxy reduces the off_rtg feature, not actual points directly.
  - If injury data is missing/unavailable, no adjustment is made (fallback).

These are proxy estimates — true impact varies by player and system.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.logging import logger

# Assumed PPP impact per top-3 player missing (3% of team offensive rating)
TOP3_PPP_IMPACT_PCT: float = 0.03


class InjuryAdjuster:
    """
    Computes top-3 players per team and adjusts feature rows for injuries.

    Parameters
    ----------
    player_season_stats : DataFrame with columns
        [team_id, player_name, usage_pct, avg_minutes, season]
        (built from silver box scores)
    """

    def __init__(self, player_season_stats: Optional[pd.DataFrame] = None) -> None:
        self._stats = player_season_stats
        self._top3_cache: dict[str, list[str]] = {}

    def get_top3_players(self, team_id: str, season: str) -> list[str]:
        """
        Return the top-3 player names by usage_pct × avg_minutes.
        Falls back to empty list if data unavailable.
        """
        cache_key = f"{team_id}_{season}"
        if cache_key in self._top3_cache:
            return self._top3_cache[cache_key]

        if self._stats is None or self._stats.empty:
            return []

        team_stats = self._stats[
            (self._stats["team_id"] == team_id)
            & (self._stats["season"] == season)
        ].copy()

        if team_stats.empty:
            return []

        if "usage_pct" not in team_stats.columns or "avg_minutes" not in team_stats.columns:
            logger.debug("Missing usage_pct or avg_minutes for team={}", team_id)
            return []

        team_stats["impact"] = team_stats["usage_pct"] * team_stats["avg_minutes"]
        top3 = team_stats.nlargest(3, "impact")["player_name"].tolist()
        self._top3_cache[cache_key] = top3
        return top3

    def adjust_off_rtg(
        self,
        base_off_rtg: float,
        team_id: str,
        season: str,
        unavailable_players: list[str],
    ) -> float:
        """
        Apply PPP-proxy reduction to offensive rating.

        Each top-3 player that is unavailable reduces off_rtg by TOP3_PPP_IMPACT_PCT.
        Returns base_off_rtg unchanged if no injury data or no top-3 players missing.
        """
        if not unavailable_players:
            return base_off_rtg

        top3 = self.get_top3_players(team_id, season)
        if not top3:
            logger.debug(
                "No top-3 data for team={} season={} — no injury adjustment applied",
                team_id, season,
            )
            return base_off_rtg

        # Case-insensitive matching
        unavail_lower = {p.lower() for p in unavailable_players}
        top3_missing = sum(1 for p in top3 if p.lower() in unavail_lower)

        if top3_missing == 0:
            return base_off_rtg

        reduction = TOP3_PPP_IMPACT_PCT * top3_missing
        adjusted = base_off_rtg * (1 - reduction)
        logger.debug(
            "Injury adjustment: team={} top3_missing={} off_rtg {:.1f} → {:.1f} ({:.0%} reduction)",
            team_id, top3_missing, base_off_rtg, adjusted, reduction,
        )
        return float(adjusted)

    def apply_adjustment(
        self,
        features_row: dict[str, Any],
        injury_data: dict[str, list[str]],
        season: str,
    ) -> dict[str, Any]:
        """
        Apply injury adjustments to a single matchup feature row.

        Parameters
        ----------
        features_row : dict of feature name → value (one game)
        injury_data  : {team_id: [unavailable_player_names]}  — may be {}
        season       : current season string

        Returns
        -------
        Adjusted features_row (same keys, modified off_rtg values)
        """
        if not injury_data:
            return features_row  # No-op fallback

        row = dict(features_row)

        home_team_id = row.get("home_team_id", "")
        away_team_id = row.get("away_team_id", "")

        # Home team adjustment
        if home_team_id and home_team_id in injury_data:
            for suffix in ("off_rtg_l5", "off_rtg_l10", "off_rtg_std"):
                col = f"home_{suffix}"
                if col in row and row[col] is not None:
                    row[col] = self.adjust_off_rtg(
                        float(row[col]), home_team_id, season,
                        injury_data[home_team_id],
                    )
            row["home_top3_avail"] = int(
                not bool(
                    set(p.lower() for p in self.get_top3_players(home_team_id, season))
                    & set(p.lower() for p in injury_data[home_team_id])
                )
            )

        # Away team adjustment
        if away_team_id and away_team_id in injury_data:
            for suffix in ("off_rtg_l5", "off_rtg_l10", "off_rtg_std"):
                col = f"away_{suffix}"
                if col in row and row[col] is not None:
                    row[col] = self.adjust_off_rtg(
                        float(row[col]), away_team_id, season,
                        injury_data[away_team_id],
                    )
            row["away_top3_avail"] = int(
                not bool(
                    set(p.lower() for p in self.get_top3_players(away_team_id, season))
                    & set(p.lower() for p in injury_data[away_team_id])
                )
            )

        return row

    @classmethod
    def from_silver(cls, season: Optional[str] = None) -> "InjuryAdjuster":
        """
        Build adjuster from silver box score data.
        Computes per-player season-to-date usage_pct and avg_minutes.
        """
        try:
            from src.pipeline.bronze_to_silver import load_silver
            bs_df = load_silver("box_scores")

            if bs_df.empty:
                logger.warning("Silver box scores empty — InjuryAdjuster has no player data")
                return cls()

            if season:
                bs_df = bs_df[bs_df.get("season", pd.Series()) == season] if "season" in bs_df.columns else bs_df

            # Aggregate per player per season
            agg = (
                bs_df.groupby(["team_id", "player_name"])
                .agg(
                    total_fga=("fga", "sum"),
                    games=("game_id", "nunique"),
                    total_minutes=("minutes", "sum"),
                )
                .reset_index()
            )
            agg["avg_minutes"] = agg["total_minutes"] / agg["games"].clip(lower=1)

            # Proxy usage_pct = team share of FGA (simple volume proxy)
            team_fga = agg.groupby("team_id")["total_fga"].transform("sum").clip(lower=1)
            agg["usage_pct"] = agg["total_fga"] / team_fga
            agg["season"] = season or "current"

            return cls(player_season_stats=agg)

        except Exception as exc:
            logger.warning("Could not build InjuryAdjuster from silver: {}", exc)
            return cls()
