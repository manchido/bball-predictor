"""
Silver → Gold layer.

Reads silver parquet tables, computes all features,
and writes the final modelling table to /data/gold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.bronze_to_silver import load_silver, save_silver
from src.pipeline.features import build_matchup_features, compute_possessions, compute_pace_per40
from src.utils.config import settings
from src.utils.logging import logger


def build_gold_table(seasons: list[str] | None = None) -> pd.DataFrame:
    """
    Full silver → gold build.

    1. Load silver box scores + schedules
    2. Compute possessions + pace
    3. Compute OffRtg / DefRtg
    4. Build rolling features (no leakage)
    5. Join odds if available
    6. Save to gold/features.parquet

    Parameters
    ----------
    seasons : if provided, filter to these seasons only

    Returns
    -------
    Gold feature DataFrame
    """
    logger.info("Starting silver → gold build")

    # -- Load silver tables --
    bs_df = load_silver("box_scores")
    sched_df = load_silver("schedules")
    odds_df = load_silver("odds")

    if bs_df.empty or sched_df.empty:
        logger.error("Silver tables empty — run bronze → silver first")
        return pd.DataFrame()

    # -- Filter seasons --
    if seasons:
        if "season" in sched_df.columns:
            sched_df = sched_df[sched_df["season"].isin(seasons)]
        if "game_id" in bs_df.columns:
            valid_ids = sched_df["game_id"].unique()
            bs_df = bs_df[bs_df["game_id"].isin(valid_ids)]

    # -- Possessions (from box score — never use listed pace) --
    bs_df["possessions"] = compute_possessions(
        bs_df["fga"], bs_df["orb"], bs_df["tov"], bs_df["fta"]
    )
    bs_df["pace_per40"] = compute_pace_per40(bs_df["possessions"], bs_df["game_minutes"])

    # -- Aggregate to team-game level --
    team_game = (
        bs_df.groupby(["game_id", "team_id", "team_side"])
        .agg(
            points=("points_norm", "sum"),
            possessions=("poss_norm", "sum"),
            pace_per40=("pace_per40", "mean"),
            fga=("fga", "sum"),
            fta=("fta", "sum"),
            game_minutes=("game_minutes", "first"),
            ot_periods=("ot_periods", "first"),
        )
        .reset_index()
    )

    # Merge schedule metadata
    sched_cols = ["game_id", "date", "league", "season", "home_team_id", "away_team_id"]
    sched_cols = [c for c in sched_cols if c in sched_df.columns]
    team_game = team_game.merge(sched_df[sched_cols].drop_duplicates("game_id"), on="game_id", how="left")
    team_game["is_home"] = team_game["team_side"] == "home"

    # Opponent points (for DefRtg proxy)
    game_pts = team_game.groupby("game_id")["points"].sum().rename("game_total")
    team_game = team_game.merge(game_pts, on="game_id")
    team_game["opp_points"] = team_game["game_total"] - team_game["points"]

    # Simple OffRtg / DefRtg per 100 possessions
    poss_safe = team_game["possessions"].clip(lower=1)
    team_game["off_rtg"] = (team_game["points"] / poss_safe) * 100
    team_game["def_rtg"] = (team_game["opp_points"] / poss_safe) * 100

    # -- Build matchup features (rolling + H2H + rest) --
    gold_df = build_matchup_features(
        team_game_df=team_game,
        schedule_df=team_game[["game_id", "date", "team_id", "is_home"]],
    )

    # -- Merge bookmaker odds --
    if not odds_df.empty and "game_id" in odds_df.columns:
        odds_cols = ["game_id", "book_total", "book_spread", "odds_source"]
        odds_cols = [c for c in odds_cols if c in odds_df.columns]
        # Take most recent odds row per game
        latest_odds = odds_df.sort_values("date").drop_duplicates("game_id", keep="last")
        gold_df = gold_df.merge(latest_odds[odds_cols], on="game_id", how="left")

    # -- Save --
    gold_path = settings.gold_dir / "features.parquet"
    settings.gold_dir.mkdir(parents=True, exist_ok=True)
    gold_df.to_parquet(gold_path, index=False, engine="pyarrow")
    logger.info("Gold table saved: {} rows, {} cols → {}", len(gold_df), len(gold_df.columns), gold_path)

    return gold_df


def load_gold() -> pd.DataFrame:
    """Load the gold feature table."""
    gold_path = settings.gold_dir / "features.parquet"
    if not gold_path.exists():
        logger.error("Gold table not found at {}. Run build_gold_table() first.", gold_path)
        return pd.DataFrame()
    return pd.read_parquet(gold_path)
