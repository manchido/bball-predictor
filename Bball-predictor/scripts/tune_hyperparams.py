"""
Hyperparameter tuning with Optuna.

Searches XGBoost + LightGBM params to minimise validation-set MAE.
Uses the same train/val temporal split as the main trainer.

Usage:
    source .venv/bin/activate
    python scripts/tune_hyperparams.py [--trials 100] [--timeout 3600]

Output:
    models/best_params.json  — auto-loaded by trainer.py on next `python cli.py train`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import typer
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Make sure project root is on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.features import FEATURE_COLS, TARGET_HOME, TARGET_AWAY
from src.pipeline.silver_to_gold import load_gold
from src.utils.config import settings
from src.utils.logging import setup_logging, logger

TRAIN_SEASONS   = ["2023-24", "2024", "2024-25", "2025"]
CURRENT_SEASONS = ["2025-26", "2026"]
VAL_CUTOFF      = pd.Timestamp("2026-01-01")  # first half of current season → val
SEED = settings.seed


def _split(gold_df):
    gold_df = gold_df.copy()
    gold_df["date"] = pd.to_datetime(gold_df["date"])
    train   = gold_df[gold_df["season"].isin(TRAIN_SEASONS)]
    current = gold_df[gold_df["season"].isin(CURRENT_SEASONS)]
    val     = current[current["date"] < VAL_CUTOFF]
    return train, val


def _xy(df):
    available = [c for c in FEATURE_COLS if c in df.columns]
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    X      = df[FEATURE_COLS].fillna(0).values.astype("float32")
    y_home = df[TARGET_HOME].fillna(0).values.astype("float32")
    y_away = df[TARGET_AWAY].fillna(0).values.astype("float32")
    return X, y_home, y_away


def _fit_and_score(xgb_p: dict, lgbm_p: dict, rf_p: dict,
                   X_tr, y_h_tr, y_a_tr,
                   X_vl, y_h_vl, y_a_vl) -> float:
    """
    Train a stacked ensemble with given params.

    Returns a combined score:
        per-side MAE  +  calibration penalty

    Calibration penalty = 15 * |raw_coverage - 0.80| per side.
    This prevents Optuna from finding params that nail MAE but produce
    poorly-calibrated intervals (the problem from round 1 of tuning).
    """
    total_mae = 0.0
    total_cal_penalty = 0.0

    for y_tr, y_vl in [(y_h_tr, y_h_vl), (y_a_tr, y_a_vl)]:
        xgb  = XGBRegressor(**xgb_p)
        lgbm = LGBMRegressor(**lgbm_p)
        rf   = RandomForestRegressor(**rf_p)

        xgb.fit(X_tr, y_tr)
        lgbm.fit(X_tr, y_tr)
        rf.fit(X_tr, y_tr)

        # Point prediction via stacked meta
        val_stack = np.column_stack([
            xgb.predict(X_vl),
            lgbm.predict(X_vl),
            rf.predict(X_vl),
        ])
        meta = Ridge(alpha=1.0)
        meta.fit(val_stack, y_vl)
        pred = meta.predict(val_stack)
        total_mae += mean_absolute_error(y_vl, pred)

        # Interval calibration: quantile LGBMs on train, evaluated on val
        q10 = LGBMRegressor(objective="quantile", alpha=0.10, **lgbm_p)
        q90 = LGBMRegressor(objective="quantile", alpha=0.90, **lgbm_p)
        q10.fit(X_tr, y_tr)
        q90.fit(X_tr, y_tr)
        p10 = q10.predict(X_vl)
        p90 = q90.predict(X_vl)

        # Raw coverage on val — penalise deviation from 80% target
        coverage = float(np.mean((y_vl >= p10) & (y_vl <= p90)))
        total_cal_penalty += abs(coverage - 0.80)

    mae_score = total_mae / 2
    cal_score = total_cal_penalty / 2  # avg per side

    # Weight: 1% calibration miss ≈ 0.15 MAE penalty
    return mae_score + 15.0 * cal_score


def make_objective(X_tr, y_h_tr, y_a_tr, X_vl, y_h_vl, y_a_vl):
    def objective(trial: optuna.Trial) -> float:
        # ── XGBoost params ───────────────────────────────────────────────
        xgb_p = dict(
            n_estimators      = trial.suggest_int("xgb_n_est",    200, 900),
            learning_rate     = trial.suggest_float("xgb_lr",     0.01, 0.15, log=True),
            max_depth         = trial.suggest_int("xgb_depth",    3, 7),
            subsample         = trial.suggest_float("xgb_sub",    0.55, 1.0),
            colsample_bytree  = trial.suggest_float("xgb_col",    0.45, 1.0),
            min_child_weight  = trial.suggest_int("xgb_mcw",      1, 12),
            reg_alpha         = trial.suggest_float("xgb_alpha",  0.0, 4.0),
            reg_lambda        = trial.suggest_float("xgb_lambda", 0.0, 4.0),
            random_state=SEED, n_jobs=-1, verbosity=0,
        )

        # ── LightGBM params ──────────────────────────────────────────────
        lgbm_p = dict(
            n_estimators      = trial.suggest_int("lgbm_n_est",   200, 900),
            learning_rate     = trial.suggest_float("lgbm_lr",    0.01, 0.15, log=True),
            num_leaves        = trial.suggest_int("lgbm_leaves",  20, 110),
            subsample         = trial.suggest_float("lgbm_sub",   0.55, 1.0),
            colsample_bytree  = trial.suggest_float("lgbm_col",   0.45, 1.0),
            min_child_samples = trial.suggest_int("lgbm_mcs",     5, 40),
            reg_alpha         = trial.suggest_float("lgbm_alpha", 0.0, 4.0),
            reg_lambda        = trial.suggest_float("lgbm_lambda",0.0, 4.0),
            random_state=SEED, n_jobs=-1, verbose=-1,
        )

        # ── RandomForest params ──────────────────────────────────────────
        rf_p = dict(
            n_estimators    = trial.suggest_int("rf_n_est",   100, 500),
            max_features    = trial.suggest_float("rf_feat",  0.3, 0.9),
            min_samples_leaf= trial.suggest_int("rf_msl",     2, 15),
            random_state=SEED, n_jobs=-1,
        )

        return _fit_and_score(xgb_p, lgbm_p, rf_p,
                              X_tr, y_h_tr, y_a_tr,
                              X_vl, y_h_vl, y_a_vl)

    return objective


app = typer.Typer(add_completion=False)


@app.command()
def main(
    trials:  int = typer.Option(100,  "--trials",  help="Number of Optuna trials"),
    timeout: int = typer.Option(3600, "--timeout", help="Max wall-clock seconds"),
    jobs:    int = typer.Option(1,    "--jobs",    help="Parallel Optuna workers (1=serial)"),
) -> None:
    """Tune XGBoost + LightGBM + RF hyperparameters with Optuna."""
    setup_logging(log_level="WARNING", log_dir=settings.logs_dir)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Loading gold table...")
    gold_df = load_gold()
    if gold_df.empty:
        typer.echo("Gold table empty — run build-gold first.", err=True)
        raise typer.Exit(1)

    gold_df["season"] = gold_df["season"].astype(str)
    train_df, val_df = _split(gold_df)
    print(f"Train: {len(train_df)} rows  |  Val: {len(val_df)} rows")

    if train_df.empty or val_df.empty:
        typer.echo("Split produced empty set — check season labels.", err=True)
        raise typer.Exit(1)

    X_tr, y_h_tr, y_a_tr = _xy(train_df.copy())
    X_vl, y_h_vl, y_a_vl = _xy(val_df.copy())

    study = optuna.create_study(
        direction="minimize",
        study_name="bball_hyperparams",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    # Warm-start: seed with the previously-found best params so we explore
    # from a known-good region rather than starting cold.
    out_path = Path(settings.models_dir) / "best_params.json"
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        xp = prev.get("xgb_params", {})
        lp = prev.get("lgbm_params", {})
        rp = prev.get("rf_params", {})
        seed_params = {
            "xgb_n_est":    xp.get("n_estimators", 400),
            "xgb_lr":       xp.get("learning_rate", 0.05),
            "xgb_depth":    xp.get("max_depth", 5),
            "xgb_sub":      xp.get("subsample", 0.8),
            "xgb_col":      xp.get("colsample_bytree", 0.8),
            "xgb_mcw":      xp.get("min_child_weight", 3),
            "xgb_alpha":    xp.get("reg_alpha", 0.1),
            "xgb_lambda":   xp.get("reg_lambda", 1.0),
            "lgbm_n_est":   lp.get("n_estimators", 400),
            "lgbm_lr":      lp.get("learning_rate", 0.05),
            "lgbm_leaves":  lp.get("num_leaves", 63),
            "lgbm_sub":     lp.get("subsample", 0.8),
            "lgbm_col":     lp.get("colsample_bytree", 0.8),
            "lgbm_mcs":     lp.get("min_child_samples", 10),
            "lgbm_alpha":   lp.get("reg_alpha", 0.1),
            "lgbm_lambda":  lp.get("reg_lambda", 1.0),
            "rf_n_est":     rp.get("n_estimators", 300),
            "rf_feat":      rp.get("max_features", 0.7),
            "rf_msl":       rp.get("min_samples_leaf", 5),
        }
        study.enqueue_trial(seed_params)
        print(f"Warm-starting from existing best_params.json")

    objective = make_objective(X_tr, y_h_tr, y_a_tr, X_vl, y_h_vl, y_a_vl)

    print(f"Running {trials} trials (timeout={timeout}s)...")
    study.optimize(
        objective,
        n_trials=trials,
        timeout=timeout,
        n_jobs=jobs,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest val MAE: {best.value:.4f}  (trial #{best.number})")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Reconstruct param dicts from flat Optuna namespace
    p = best.params
    best_params = {
        "best_val_mae": best.value,
        "xgb_params": {
            "n_estimators":     p["xgb_n_est"],
            "learning_rate":    p["xgb_lr"],
            "max_depth":        p["xgb_depth"],
            "subsample":        p["xgb_sub"],
            "colsample_bytree": p["xgb_col"],
            "min_child_weight": p["xgb_mcw"],
            "reg_alpha":        p["xgb_alpha"],
            "reg_lambda":       p["xgb_lambda"],
            "random_state": SEED, "n_jobs": -1, "verbosity": 0,
        },
        "lgbm_params": {
            "n_estimators":      p["lgbm_n_est"],
            "learning_rate":     p["lgbm_lr"],
            "num_leaves":        p["lgbm_leaves"],
            "subsample":         p["lgbm_sub"],
            "colsample_bytree":  p["lgbm_col"],
            "min_child_samples": p["lgbm_mcs"],
            "reg_alpha":         p["lgbm_alpha"],
            "reg_lambda":        p["lgbm_lambda"],
            "random_state": SEED, "n_jobs": -1, "verbose": -1,
        },
        "rf_params": {
            "n_estimators":     p["rf_n_est"],
            "max_features":     p["rf_feat"],
            "min_samples_leaf": p["rf_msl"],
            "random_state": SEED, "n_jobs": -1,
        },
    }

    out_path = Path(settings.models_dir) / "best_params.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(best_params, indent=2))
    print(f"\nSaved → {out_path}")
    print("Run `python cli.py train` to retrain with the tuned params.")


if __name__ == "__main__":
    app()
