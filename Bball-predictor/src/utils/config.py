from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Bookmaker / odds
    odds_api_key: str = ""
    odds_source: str = "the-odds-api"  # stamped on every tracker row

    # Scrapers
    realgm_base_url: str = "https://basketball.realgm.com"
    realgm_request_delay: float = 2.0  # seconds between requests

    # Reproducibility
    seed: int = 42

    # Paths (resolved relative to project root at runtime)
    data_dir: str = "data"
    models_dir: str = "models"
    tracking_dir: str = "tracking"
    logs_dir: str = "logs"

    # Timezone for all timestamps
    timezone: str = "America/St_Johns"

    # Logging
    log_level: str = "INFO"

    @property
    def bronze_dir(self) -> Path:
        return Path(self.data_dir) / "bronze"

    @property
    def silver_dir(self) -> Path:
        return Path(self.data_dir) / "silver"

    @property
    def gold_dir(self) -> Path:
        return Path(self.data_dir) / "gold"

    @property
    def tracking_path(self) -> Path:
        return Path(self.tracking_dir) / "daily_tracker.csv"

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        dirs = [
            self.bronze_dir / "realgm" / "schedules",
            self.bronze_dir / "realgm" / "box_scores",
            self.bronze_dir / "realgm" / "team_stats",
            self.bronze_dir / "odds",
            self.silver_dir / "schedules",
            self.silver_dir / "box_scores",
            self.silver_dir / "odds",
            self.gold_dir,
            Path(self.models_dir),
            Path(self.tracking_dir),
            Path(self.logs_dir),
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# Singleton — import and use `settings` directly
settings = Settings()
