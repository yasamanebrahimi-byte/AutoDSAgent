"""Application logging helpers."""

from __future__ import annotations

import logging
import sys
from typing import Any

from app.backend.config import Settings, settings


LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s "
    "environment=%(autods_environment)s %(message)s"
)


class AutoDSLogFilter(logging.Filter):
    """Attach project metadata to log records."""

    def __init__(self, environment: str) -> None:
        super().__init__()
        self.environment = environment

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "autods_environment"):
            record.autods_environment = self.environment
        return True


def configure_logging(active_settings: Settings | None = None) -> None:
    """Configure process logging once."""

    selected = active_settings or settings
    root_logger = logging.getLogger()
    level = getattr(logging, selected.log_level.upper(), logging.INFO)
    root_logger.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT)
    existing_stream_handlers = [
        handler
        for handler in root_logger.handlers
        if getattr(handler, "_autods_handler", False)
    ]
    if not existing_stream_handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler._autods_handler = True  # type: ignore[attr-defined]
        root_logger.addHandler(handler)
        existing_stream_handlers.append(handler)

    for handler in existing_stream_handlers:
        handler.setFormatter(formatter)
        handler.setLevel(level)
        if not any(isinstance(item, AutoDSLogFilter) for item in handler.filters):
            handler.addFilter(AutoDSLogFilter(selected.environment))


def get_logger(name: str) -> logging.Logger:
    """Return a project logger."""

    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    **details: Any,
) -> None:
    """Log a message with lightweight structured key-value details."""

    if details:
        detail_text = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
        logger.log(level, "%s %s", message, detail_text)
    else:
        logger.log(level, "%s", message)
