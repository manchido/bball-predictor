"""
Feature engineering: Silver → Gold.

Strictly pre-game features only — all rolling windows use .shift(1)
so only games BEFORE the current one inform the feature.

Feature categories:
  1. Team offense:   off_rtg, pace, pts_scored (L5, L10, STD)
  2. Team defense:   def_rtg, pts_allowed, opp_pace (L5, L10, STD)
  3. Head-to-head:  last-3 meetings total pts mean
  4. Rest/travel:   days_rest, consecutive_away
  5. Injury proxy:  top3_avail (0/1)
  6. Context:       league_id, season, week_of_season, is_home
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from src.utils.logging import logger


# ---------------------------------------------------------------------------
# Possessions (Oliver formula) — computed from box score, not listed pace
# ---------------------------------------------------------------------------

def compute_possessions(fga: pd.Series, orb: pd.Series, tov: pd.Series, fta: pd.Series) -> pd.Series:
    """POSS = FGA - ORB + TOV + 0.44 * FTA."""
    return fga - orb + tov + 0.44 * fta


def compute_pace_per40(possessions: pd.Series, game_minutes: pd.Series) -> pd.Series:
    """pace_per40 = (POSS / game_minutes) * 40."""
    return (possessions / game_minutes.clip(lower=1)) * 40


# ---------------------------------------------------------------------------
# Rolling feature builder
# ---------------------------------------------------------------------------

def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Shifted rolling mean — strictly pre-game."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def _cumulative_mean(series: pd.Series) -> pd.Series:
    """Season-to-date mean — strictly pre-game."""
    return series.shift(1).expanding(min_periods=1).mean()


def build_team_rolling_features(team_game_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a team-game DataFrame (one row per team per game, sorted by date),
    compute rolling offensive/defensive features with no leakage.

    Input columns expected:
      team_id, game_id, date, is_home, points (scored by this team),
      opp_points (points allowed), possessions, pace_per40, off_rtg, def_rtg

    Returns the same DataFrame with additional feature columns.
    """
    df = team_game_df.sort_values(["team_id", "date"]).copy()

    feature_cols = {
        "off_rtg": ["l5", "l10", "std"],
        "def_rtg": ["l5", "l10", "std"],
        "pace_per40": ["l5", "l10", "std"],
        "points": ["l5", "l10", "std"],  # pts scored
        "opp_points": ["l5", "l10", "std"],  # pts allowed
        "possessions": ["l5", "l10"],
    }

    for col, windows in feature_cols.items():
        if col not in df.columns:
            continue
        grp = df.groupby("team_id")[col]
        for w in windows:
            if w == "std":
                df[f"{col}_std"] = grp.transform(_cumulative_mean)
            elif w == "l5":
                df[f"{col}_l5"] = grp.transform(lambda s: _rolling_mean(s, 5))
            elif w == "l10":
                df[f"{col}_l10"] = grp.transform(lambda s: _rolling_mean(s, 10))

    # Home/Away specific rolling (pts scored)
    for side, flag in [("home", True), ("away", False)]:
        mask = df["is_home"] == flag
        subset = df[mask].copy()
        if subset.empty:
            continue
        grp = subset.groupby("team_id")["points"]
        df.loc[mask, f"pts_scored_{side}_l5"] = grp.transform(lambda s: _rolling_mean(s, 5))
        df.loc[mask, f"pts_scored_{side}_l10"] = grp.transform(lambda s: _rolling_mean(s, 10))

    return df


def build_rest_features(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute days_rest and consecutive_away for each team-game.

    Input: schedule with columns [game_id, date, team_id, is_home]
    """
    df = schedule_df.sort_values(["team_id", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["prev_date"] = df.groupby("team_id")["date"].shift(1)
    df["days_rest"] = (df["date"] - df["prev_date"]).dt.days.fillna(7).clip(upper=14)

    # Consecutive away games
    df["away_flag"] = (~df["is_home"]).astype(int)
    df["cons_away"] = (
        df.groupby("team_id")["away_flag"]
        .transform(lambda s: s.groupby((s != s.shift()).cumsum()).cumcount())
    )
    return df


def build_h2h_features(team_game_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute head-to-head total points mean from last 3 meetings.

    Input must include: game_id, home_team_id, away_team_id, game_total, date
    """
    df = team_game_df.drop_duplicates("game_id").copy()
    df = df.sort_values("date")

    h2h_means: dict[str, float] = {}
    for _, row in df.iterrows():
        pair = frozenset([row["home_team_id"], row["away_team_id"]])
        past = df[
            df.apply(
                lambda r: frozenset([r["home_team_id"], r["away_team_id"]]) == pair, axis=1
            )
            & (df["date"] < row["date"])
        ]
        if len(past) > 0:
            h2h_means[row["game_id"]] = past["game_total"].tail(3).mean()
        else:
            h2h_means[row["game_id"]] = np.nan

    df["h2h_total_l3"] = df["game_id"].map(h2h_means)
    return df[["game_id", "h2h_total_l3"]]


# ---------------------------------------------------------------------------
# Main feature build entry point
# ---------------------------------------------------------------------------

def build_matchup_features(
    team_game_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    injury_availability: Optional[dict[str, bool]] = None,
) -> pd.DataFrame:
    """
    Build the complete matchup-level feature table for modelling.

    Each row = one game.
    Output columns are split into home_* and away_* perspectives.

    Parameters
    ----------
    team_game_df : team-level stats (one row per team per game)
    schedule_df  : schedule with date/team/home columns
    injury_availability : {team_id: top3_available (bool)} — optional

    Returns
    -------
    DataFrame with one row per game, ready for the model.
    """
    # Ensure date types
    team_game_df = team_game_df.copy()
    team_game_df["date"] = pd.to_datetime(team_game_df["date"])

    # Rolling features
    team_game_df = build_team_rolling_features(team_game_df)

    # Rest features
    rest_df = build_rest_features(
        schedule_df[["game_id", "date", "team_id", "is_home"]]
    )
    team_game_df = team_game_df.merge(
        rest_df[["game_id", "team_id", "days_rest", "cons_away"]],
        on=["game_id", "team_id"],
        how="left",
    )

    # Pivot to game-level (home vs away)
    home = team_game_df[team_game_df["is_home"]].copy()
    away = team_game_df[~team_game_df["is_home"]].copy()

    rolling_cols = [
        c for c in team_game_df.columns
        if any(c.endswith(sfx) for sfx in ("_l5", "_l10", "_std"))
    ] + ["days_rest", "cons_away", "possessions", "pace_per40", "game_minutes", "ot_periods"]

    home_feat = home[["game_id", "team_id", "points", "opp_points"] + rolling_cols].copy()
    home_feat.columns = (
        ["game_id", "home_team_id", "home_points", "home_opp_points"]
        + [f"home_{c}" for c in rolling_cols]
    )

    away_feat = away[["game_id", "team_id", "points", "opp_points"] + rolling_cols].copy()
    away_feat.columns = (
        ["game_id", "away_team_id", "away_points", "away_opp_points"]
        + [f"away_{c}" for c in rolling_cols]
    )

    game_df = home_feat.merge(away_feat, on="game_id", how="inner")
    game_df["game_total"] = game_df["home_points"] + game_df["away_points"]

    # Head-to-head
    h2h_df = build_h2h_features(
        team_game_df.drop_duplicates("game_id")[
            ["game_id", "home_team_id", "away_team_id", "game_total", "date"]
        ]
        if "home_team_id" in team_game_df.columns
        else team_game_df.drop_duplicates("game_id")
    )
    game_df = game_df.merge(h2h_df, on="game_id", how="left")

    # League/season context
    if "league" in team_game_df.columns:
        league_map = team_game_df.drop_duplicates("game_id").set_index("game_id")["league"]
        game_df["league"] = game_df["game_id"].map(league_map)
        game_df["league_id"] = pd.Categorical(game_df["league"]).codes

    if "season" in team_game_df.columns:
        season_map = team_game_df.drop_duplicates("game_id").set_index("game_id")["season"]
        game_df["season"] = game_df["game_id"].map(season_map)

    if "date" in team_game_df.columns:
        date_map = team_game_df.drop_duplicates("game_id").set_index("game_id")["date"]
        game_df["date"] = game_df["game_id"].map(date_map)
        game_df["week_of_season"] = pd.to_datetime(game_df["date"]).dt.isocalendar().week

    # Injury proxy (top-3 availability)
    game_df["home_top3_avail"] = 1
    game_df["away_top3_avail"] = 1
    if injury_availability:
        game_df["home_top3_avail"] = game_df["home_team_id"].map(
            lambda tid: int(injury_availability.get(tid, True))
        )
        game_df["away_top3_avail"] = game_df["away_team_id"].map(
            lambda tid: int(injury_availability.get(tid, True))
        )

    logger.info("Built matchup features: {} games, {} columns", len(game_df), len(game_df.columns))
    return game_df


# ---------------------------------------------------------------------------
# Feature column list (used by trainer to select X)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Offense rolling
    "home_off_rtg_l5", "home_off_rtg_l10", "home_off_rtg_std",
    "away_off_rtg_l5", "away_off_rtg_l10", "away_off_rtg_std",
    # Defense rolling
    "home_def_rtg_l5", "home_def_rtg_l10", "home_def_rtg_std",
    "away_def_rtg_l5", "away_def_rtg_l10", "away_def_rtg_std",
    # Pace
    "home_pace_per40_l5", "home_pace_per40_l10", "home_pace_per40_std",
    "away_pace_per40_l5", "away_pace_per40_l10", "away_pace_per40_std",
    # Points scored
    "home_points_l5", "home_points_l10", "home_points_std",
    "away_points_l5", "away_points_l10", "away_points_std",
    # Points allowed
    "home_opp_points_l5", "home_opp_points_l10", "home_opp_points_std",
    "away_opp_points_l5", "away_opp_points_l10", "away_opp_points_std",
    # Rest + travel
    "home_days_rest", "away_days_rest",
    "home_cons_away", "away_cons_away",
    # H2H
    "h2h_total_l3",
    # Injury
    "home_top3_avail", "away_top3_avail",
    # Context
    "league_id", "week_of_season",
]

TARGET_HOME = "home_points"
TARGET_AWAY = "away_points"
