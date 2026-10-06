"""Application factory and Uvicorn entry point."""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from ugsl_ai_coach.api.routes.health import router as health_router
from ugsl_ai_coach.core.config import Settings
from ugsl_ai_coach.core.logging import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(app.state.settings.log_level)
    logger.info("Service starting")
    try:
        yield
    finally:
        logger.info("Service stopping")


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="UgSL AI Practice Coach",
        version="0.1.0",
        description="Service foundation for future evidence-based UgSL practice coaching.",
        lifespan=lifespan,
    )
    app.state.settings = settings if settings is not None else Settings()
    app.include_router(health_router, prefix="/api/v1")
    return app


app = create_app()
