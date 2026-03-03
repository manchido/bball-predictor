"""POST /train — trigger model training as a background task."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from src.api.schemas import TrainRequest, TrainResponse, TrainStatusResponse
from src.utils.config import settings
from src.utils.logging import logger

router = APIRouter(tags=["training"])

# In-process job registry (sufficient for single-instance deployments)
_JOBS: dict[str, dict] = {}


def _run_training(job_id: str, request: TrainRequest) -> None:
    """Synchronous training function executed in a thread."""
    _JOBS[job_id]["status"] = "running"
    try:
        from src.pipeline.silver_to_gold import build_gold_table
        from src.models.trainer import train

        if request.force_rebuild_gold:
            logger.info("[{}] Rebuilding gold table", job_id)
            build_gold_table(seasons=request.seasons)

        logger.info("[{}] Starting training", job_id)
        metrics = train()
        _JOBS[job_id]["status"] = "done"
        _JOBS[job_id]["metrics"] = metrics
        logger.info("[{}] Training complete: {}", job_id, metrics)
    except Exception as exc:
        logger.exception("[{}] Training failed", job_id)
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["error"] = str(exc)


@router.post("", response_model=TrainResponse)
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks) -> TrainResponse:
    """
    Start a training job in the background.
    Returns a job_id to poll for status.
    """
    job_id = str(uuid.uuid4())[:8]
    _JOBS[job_id] = {"status": "started", "metrics": None, "error": None}

    loop = asyncio.get_event_loop()
    background_tasks.add_task(
        loop.run_in_executor, None, _run_training, job_id, request
    )

    logger.info("Training job {} enqueued", job_id)
    return TrainResponse(job_id=job_id, status="started", message="Training started in background")


@router.get("/{job_id}", response_model=TrainStatusResponse)
async def get_training_status(job_id: str) -> TrainStatusResponse:
    """Poll a training job by job_id."""
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    job = _JOBS[job_id]
    return TrainStatusResponse(
        job_id=job_id,
        status=job["status"],
        metrics=job.get("metrics"),
        error=job.get("error"),
    )
