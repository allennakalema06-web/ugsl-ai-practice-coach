"""Process health endpoint; no external dependency checks."""

import logging
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    logger.debug("Health check served")
    return HealthResponse(service=request.app.state.settings.service_name)
