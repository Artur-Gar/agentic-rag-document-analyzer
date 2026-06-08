from pathlib import Path

from loguru import logger

from backend.settings import ROOT_DIR


LOG_DIR = ROOT_DIR / "backend_data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.add(
    Path(LOG_DIR / "app.log"),
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)
