import sys
from pathlib import Path
from loguru import logger


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """Configure loguru for console + rotating file output."""
    logger.remove()

    # Console — human-readable
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
        colorize=True,
    )

    # Rotating file — structured for debugging
    log_path = Path(log_dir) / "bball_predictor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        enqueue=True,
    )

    logger.info("Logging initialised — level={}, file={}", log_level, log_path)


# Re-export logger so callers just do: from src.utils.logging import logger
__all__ = ["logger", "setup_logging"]
