"""
Stacked ensemble model for basketball totals prediction.

Architecture:
  Base layer:  XGBoost + LightGBM (mean) + RandomForest
  Meta layer:  Ridge regression on base layer OOF predictions

Two separate ensembles trained:
  - home_model: predicts home_team_points
  - away_model: predicts away_team_points
  game_total_mean = home_pred + away_pred

Intervals:
  p10/p90 via LightGBM quantile regression (α=0.1, α=0.9)
  Combined with XGBoost bootstrap residuals for robustness.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from src.utils.config import settings
from src.utils.logging import logger

SEED = settings.seed


@dataclass
class PredictionResult:
    game_id: str
    home_pred_mean: float
    away_pred_mean: float
    total_mean: float
    total_p10: float
    total_p50: float
    total_p90: float
    confidence: float  # width-based: 1 - (p90-p10)/total_mean


@dataclass
class StackedEnsemble:
    """
    Stacked ensemble for one target (home or away points).

    Includes quantile LGBMs for interval estimation.
    """
    seed: int = SEED
    n_estimators_xgb: int = 400
    n_estimators_lgbm: int = 400
    n_estimators_rf: int = 300

    # Base models
    xgb: XGBRegressor = field(init=False)
    lgbm_mean: LGBMRegressor = field(init=False)
    lgbm_q10: LGBMRegressor = field(init=False)
    lgbm_q90: LGBMRegressor = field(init=False)
    rf: RandomForestRegressor = field(init=False)

    # Meta model
    meta: Ridge = field(init=False)
    _fitted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.xgb = XGBRegressor(
            n_estimators=self.n_estimators_xgb,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.seed,
            n_jobs=-1,
            verbosity=0,
        )
        self.lgbm_mean = LGBMRegressor(
            n_estimators=self.n_estimators_lgbm,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
        )
        self.lgbm_q10 = LGBMRegressor(
            objective="quantile",
            alpha=0.10,
            n_estimators=self.n_estimators_lgbm,
            learning_rate=0.05,
            num_leaves=63,
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
        )
        self.lgbm_q90 = LGBMRegressor(
            objective="quantile",
            alpha=0.90,
            n_estimators=self.n_estimators_lgbm,
            learning_rate=0.05,
            num_leaves=63,
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
        )
        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators_rf,
            max_features=0.7,
            min_samples_leaf=5,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.meta = Ridge(alpha=1.0)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "StackedEnsemble":
        """
        Train base models on train fold, fit meta on val fold OOF preds.
        Also fits quantile models on full train set.
        """
        logger.info("Fitting base models on {} train samples", len(X_train))

        # Fit base models on train
        self.xgb.fit(X_train, y_train)
        self.lgbm_mean.fit(X_train, y_train)
        self.rf.fit(X_train, y_train)

        # Quantile models on full train
        self.lgbm_q10.fit(X_train, y_train)
        self.lgbm_q90.fit(X_train, y_train)

        # OOF predictions on val for meta-model training
        val_preds = np.column_stack([
            self.xgb.predict(X_val),
            self.lgbm_mean.predict(X_val),
            self.rf.predict(X_val),
        ])

        logger.info("Fitting Ridge meta-model on {} val samples", len(X_val))
        self.meta.fit(val_preds, y_val)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (mean_pred, p10_pred, p90_pred).
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        base = np.column_stack([
            self.xgb.predict(X),
            self.lgbm_mean.predict(X),
            self.rf.predict(X),
        ])
        mean_pred = self.meta.predict(base)
        p10_pred = self.lgbm_q10.predict(X)
        p90_pred = self.lgbm_q90.predict(X)

        # Ensure p10 < mean < p90 (correct ordering)
        p10_pred = np.minimum(p10_pred, mean_pred)
        p90_pred = np.maximum(p90_pred, mean_pred)

        return mean_pred, p10_pred, p90_pred


class BballEnsemble:
    """
    Top-level ensemble: trains home + away models, produces game-total predictions.
    """

    def __init__(self) -> None:
        self.home_model = StackedEnsemble(seed=SEED)
        self.away_model = StackedEnsemble(seed=SEED)
        self._calibration_scale: float = 1.0  # set during training
        self._league_home_bias: dict[str, float] = {}  # per-league mean home residual
        self._league_away_bias: dict[str, float] = {}  # per-league mean away residual
        self._fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_home_train: np.ndarray,
        y_away_train: np.ndarray,
        X_val: np.ndarray,
        y_home_val: np.ndarray,
        y_away_val: np.ndarray,
    ) -> "BballEnsemble":
        logger.info("Training home model")
        self.home_model.fit(X_train, y_home_train, X_val, y_home_val)
        logger.info("Training away model")
        self.away_model.fit(X_train, y_away_train, X_val, y_away_val)
        self._fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        game_ids: Optional[list[str]] = None,
        leagues: Optional[list[str]] = None,
    ) -> list[PredictionResult]:
        if not self._fitted:
            raise RuntimeError("Ensemble not fitted.")

        home_mean, home_p10, home_p90 = self.home_model.predict(X)
        away_mean, away_p10, away_p90 = self.away_model.predict(X)

        scale = self._calibration_scale
        results: list[PredictionResult] = []

        for i in range(len(home_mean)):
            # Per-league bias correction (zero if league unknown or bias not fitted)
            if leagues and self._league_home_bias:
                lg = leagues[i]
                home_bias = self._league_home_bias.get(lg, 0.0)
                away_bias = self._league_away_bias.get(lg, 0.0)
            else:
                home_bias = away_bias = 0.0

            h_mean = float(home_mean[i]) + home_bias
            a_mean = float(away_mean[i]) + away_bias
            total_mean = h_mean + a_mean

            # Intervals: shift by total bias, keep width driven by calibration scale
            raw_p10 = float(home_p10[i] + away_p10[i])
            raw_p90 = float(home_p90[i] + away_p90[i])
            half_width = (raw_p90 - raw_p10) / 2 * scale
            p10 = total_mean - half_width
            p90 = total_mean + half_width
            p50 = total_mean

            width = p90 - p10
            confidence = float(np.clip(1 - width / max(total_mean, 1), 0, 1))

            results.append(PredictionResult(
                game_id=game_ids[i] if game_ids else str(i),
                home_pred_mean=h_mean,
                away_pred_mean=a_mean,
                total_mean=total_mean,
                total_p10=p10,
                total_p50=p50,
                total_p90=p90,
                confidence=confidence,
            ))

        return results

    def save(self, path: Optional[Path] = None) -> Path:
        save_path = path or (Path(settings.models_dir) / "ensemble.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        logger.info("Ensemble saved → {}", save_path)
        return save_path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "BballEnsemble":
        load_path = path or (Path(settings.models_dir) / "ensemble.pkl")
        if not load_path.exists():
            raise FileNotFoundError(f"No saved ensemble at {load_path}")
        model = joblib.load(load_path)
        logger.info("Ensemble loaded ← {}", load_path)
        return model
