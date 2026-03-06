"""Pydantic request/response models for the FastAPI application."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    game_id: str
    league: str
    match: str
    date: date
    book_total: Optional[float] = None
    book_spread: Optional[float] = None
    model_total_mean: float
    model_total_p10: float
    model_total_p50: float
    model_total_p90: float
    model_home_mean: float
    model_away_mean: float
    confidence: float = Field(ge=0, le=1)
    edge: Optional[float] = None          # model_mean - book_total (None if no line)
    odds_source: Optional[str] = None
    timestamp: datetime
    home_pace: Optional[float] = None          # home team pace_per40 (rolling L5) — kept for model use
    away_pace: Optional[float] = None          # away team pace_per40 (rolling L5)
    home_pts_l5: Optional[float] = None        # home team avg points scored last 5 games
    away_pts_l5: Optional[float] = None        # away team avg points scored last 5 games
    home_pts_allowed_l5: Optional[float] = None  # home team avg points allowed last 5 games
    away_pts_allowed_l5: Optional[float] = None  # away team avg points allowed last 5 games


class TodayPredictionsResponse(BaseModel):
    date: date
    leagues: list[str]
    games: list[PredictionResponse]
    model_loaded: bool
    warnings: list[str] = []


# ---------------------------------------------------------------------------
# Manual prediction
# ---------------------------------------------------------------------------

class ManualPredictRequest(BaseModel):
    league: str
    home_team: str
    away_team: str
    date: Optional[date] = None   # defaults to today if omitted


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    seasons: Optional[list[str]] = None   # If None, use all available data
    force_rebuild_gold: bool = False


class TrainResponse(BaseModel):
    job_id: str
    status: str                            # "started" | "running" | "done" | "error"
    message: str


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str
    metrics: Optional[dict[str, Any]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

class BacktestRequest(BaseModel):
    start_date: date
    end_date: date
    leagues: Optional[list[str]] = None


class BacktestMetrics(BaseModel):
    start_date: date
    end_date: date
    n_games: int
    game_total_mae: float
    game_total_rmse: float
    calibration_80pct: float
    mean_interval_width: float
    mean_edge: Optional[float] = None
    edge_hit_rate: Optional[float] = None
    leagues: list[str]


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

class TrackerRow(BaseModel):
    date: date
    league: str
    match: str
    book_total: Optional[float] = None
    book_spread: Optional[float] = None
    model_total_mean: float
    model_total_p10: float
    model_total_p90: float
    confidence: float
    edge: Optional[float] = None
    actual_total: Optional[float] = None
    error: Optional[float] = None         # actual - model_mean
    odds_source: Optional[str] = None


class TrackingResponse(BaseModel):
    rows: list[TrackerRow]
    total: int
    date_from: Optional[date] = None
    date_to: Optional[date] = None
