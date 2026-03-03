"""
FastAPI application factory.

Routes:
  POST /train             — trigger training job
  GET  /train/{job_id}    — job status
  GET  /predict/today     — today's predictions (full daily pipeline)
  GET  /predictions/{id}  — fetch stored prediction
  POST /backtest          — run backtest over date range
  GET  /tracking          — query daily tracker
  POST /tracking/update_actuals — fill in post-game actuals
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import train, predict, backtest, tracking
from src.utils.config import settings
from src.utils.logging import logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────
    setup_logging(log_level=settings.log_level, log_dir=settings.logs_dir)
    settings.ensure_dirs()
    logger.info("BballPredictor starting up")

    # Load ensemble model if available
    model_path = Path(settings.models_dir) / "ensemble.pkl"
    if model_path.exists():
        try:
            from src.models.ensemble import BballEnsemble
            model = BballEnsemble.load(model_path)
            predict.set_ensemble(model)
            logger.info("Ensemble model loaded from {}", model_path)
        except Exception as exc:
            logger.warning("Could not load model: {} — predictions unavailable until trained", exc)
    else:
        logger.warning("No saved model found at {} — POST /train to train", model_path)

    # Start APScheduler for daily pipeline
    try:
        from scheduler import start_scheduler
        start_scheduler()
    except Exception as exc:
        logger.warning("Scheduler not started: {}", exc)

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("BballPredictor shutting down")
    try:
        from scheduler import stop_scheduler
        stop_scheduler()
    except Exception:
        pass


def create_app() -> FastAPI:
    app = FastAPI(
        title="BballPredictor",
        description=(
            "International basketball game-total predictor. "
            "Covers EuroLeague, EuroCup, ACB, BSL, BBL. "
            "Bookmaker lines via The Odds API (aggregates bet365, Pinnacle, etc.)."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(train.router,     prefix="/train")
    app.include_router(predict.router,   prefix="/predict")
    app.include_router(backtest.router,  prefix="/backtest")
    app.include_router(tracking.router,  prefix="/tracking")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "model_loaded": predict._ensemble is not None}

    return app


app = create_app()
