"""Non-secret application configuration endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.backend.config import settings


router = APIRouter(tags=["config"])


@router.get("/config/status")
def get_config_status() -> dict[str, Any]:
    """Return non-secret runtime configuration status."""

    return settings.public_status()
