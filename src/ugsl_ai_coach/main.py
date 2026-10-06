"""Application factory and Uvicorn entry point."""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from ugsl_ai_coach.api.routes.contracts import router as contracts_router
from ugsl_ai_coach.api.routes.health import router as health_router
from ugsl_ai_coach.api.routes.analyses import router as analyses_router
from ugsl_ai_coach.api.errors import install_error_handlers
from ugsl_ai_coach.core.config import Settings
from ugsl_ai_coach.core.logging import configure_logging
from ugsl_ai_coach.api.routes.operations import router as operations_router
from ugsl_ai_coach.operations.metrics import Metrics
from ugsl_ai_coach.operations.http import install_operations

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
    settings = settings if settings is not None else Settings()
    docs = settings.api_docs_enabled if settings.api_docs_enabled is not None else settings.environment != 'production'
    app = FastAPI(
        title="UgSL AI Practice Coach",
        version="0.1.0",
        description="Service foundation for future evidence-based UgSL practice coaching.",
        lifespan=lifespan,
        docs_url='/docs' if docs else None,
        redoc_url='/redoc' if docs else None,
        openapi_url='/openapi.json' if docs else None,
    )
    app.state.settings = settings
    app.state.metrics = Metrics()
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(contracts_router, prefix="/api/v1")
    app.include_router(analyses_router, prefix="/api/v1")
    app.include_router(operations_router)
    install_error_handlers(app)
    install_operations(app)
    return app


app = create_app()
