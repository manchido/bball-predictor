"""
Bronze → Silver layer.

Reads raw HTML/JSON files from /data/bronze and writes
cleaned, typed parquet tables to /data/silver.

Key concerns:
- OT detection + duration normalisation
- Possessions computed from box score (NOT listed pace)
- Deduplication via game_id
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import settings
from src.utils.logging import logger


# ---------------------------------------------------------------------------
# Box scores
# ---------------------------------------------------------------------------

def process_box_scores(box_score_dicts: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of parsed box score dicts (from realgm.py) into
    a clean player-level DataFrame, then aggregate to team-game level.

    Applies:
    - OT normalisation (per-minute → scale to 40-min equivalent)
    - Possessions = FGA - ORB + TOV + 0.44 * FTA  (Oliver formula)
    """
    if not box_score_dicts:
        return pd.DataFrame()

    player_rows: list[dict[str, Any]] = []
    for box in box_score_dicts:
        game_id = box["game_id"]
        game_minutes = float(box.get("game_minutes", 40))
        ot_periods = int(box.get("ot_periods", 0))

        for p in box.get("players", []):
            player_rows.append({
                "game_id": game_id,
                "team_id": p["team_id"],
                "team_side": p["team_side"],
                "player_name": p["player_name"],
                "minutes": float(p.get("minutes", 0)),
                "points": float(p.get("points", 0)),
                "fga": float(p.get("fga", 0)),
                "orb": float(p.get("orb", 0)),
                "tov": float(p.get("tov", 0)),
                "fta": float(p.get("fta", 0)),
                "game_minutes": game_minutes,
                "ot_periods": ot_periods,
            })

    df = pd.DataFrame(player_rows)
    if df.empty:
        return df

    # OT normalisation: scale per-minute stats to 40-min equivalent
    # For players: actual_stat * (40 / game_minutes)
    ot_scale = (40.0 / df["game_minutes"]).clip(upper=1.0)  # never inflate regulation stats
    for stat in ("points", "fga", "orb", "tov", "fta"):
        df[f"{stat}_norm"] = df[stat] * ot_scale

    # Possessions per player (Oliver formula)
    df["poss"] = df["fga"] - df["orb"] + df["tov"] + 0.44 * df["fta"]
    df["poss_norm"] = df["poss"] * ot_scale

    return df


def aggregate_to_team_game(player_df: pd.DataFrame, game_meta: list[dict]) -> pd.DataFrame:
    """
    Aggregate player-level stats to team-game level.

    Columns: game_id, team_id, team_side, date, league, season,
             home_team_id, away_team_id, home_points, away_points,
             game_total, possessions, pace_per40, game_minutes, ot_periods
    """
    if player_df.empty:
        return pd.DataFrame()

    team_agg = (
        player_df.groupby(["game_id", "team_id", "team_side"])
        .agg(
            points=("points", "sum"),
            points_norm=("points_norm", "sum"),
            fga=("fga", "sum"),
            orb=("orb", "sum"),
            tov=("tov", "sum"),
            fta=("fta", "sum"),
            possessions=("poss", "sum"),
            poss_norm=("poss_norm", "sum"),
            game_minutes=("game_minutes", "first"),
            ot_periods=("ot_periods", "first"),
        )
        .reset_index()
    )

    # Pace per 40 possessions
    team_agg["pace_per40"] = (team_agg["possessions"] / team_agg["game_minutes"]) * 40

    # Merge game metadata (date, league, season, home/away team IDs)
    meta_df = pd.DataFrame(game_meta)
    if not meta_df.empty and "game_id" in meta_df.columns:
        team_agg = team_agg.merge(meta_df, on="game_id", how="left")

    return team_agg


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def process_schedules(schedule_dicts: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert raw schedule dicts to a clean DataFrame."""
    if not schedule_dicts:
        return pd.DataFrame()
    df = pd.DataFrame(schedule_dicts)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="game_id")
    return df


# ---------------------------------------------------------------------------
# Odds
# ---------------------------------------------------------------------------

def process_odds(odds_dicts: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert raw odds records to parquet-ready DataFrame."""
    if not odds_dicts:
        return pd.DataFrame()
    df = pd.DataFrame(odds_dicts)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Persist to silver layer
# ---------------------------------------------------------------------------

def save_silver(df: pd.DataFrame, table: str, partition: str = "") -> Path:
    """Save DataFrame as parquet to silver layer."""
    silver_dir = settings.silver_dir / table
    silver_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{partition}.parquet" if partition else "data.parquet"
    path = silver_dir / fname
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Saved {} rows → {}", len(df), path)
    return path


def load_silver(table: str) -> pd.DataFrame:
    """Load all parquet files from a silver table into one DataFrame."""
    silver_dir = settings.silver_dir / table
    if not silver_dir.exists():
        return pd.DataFrame()
    parts = list(silver_dir.glob("*.parquet"))
    if not parts:
        return pd.DataFrame()
    dfs = [pd.read_parquet(p) for p in parts]
    return pd.concat(dfs, ignore_index=True)
