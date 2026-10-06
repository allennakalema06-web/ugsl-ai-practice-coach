"""Coarse readiness and authenticated telemetry; no operational writes."""

from typing import Annotated
from fastapi import APIRouter, Request, Security
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST

from ugsl_ai_coach.api.auth import ServicePrincipal, StaticTokenAuthenticator, require_service
from ugsl_ai_coach.infrastructure.postgres.connection import connection_factory
from ugsl_ai_coach.infrastructure.object_store.s3 import S3MediaStore
from ugsl_ai_coach.operations.metrics import best_effort
from ugsl_ai_coach.operations.events import emit

router = APIRouter()


def get_connect(request):
    return getattr(request.app.state, 'operations_connect', None) or connection_factory(request.app.state.settings)


def validate_runtime(settings):
    # Configuration only: no boto client, credential-chain I/O, media/model download.
    StaticTokenAuthenticator(settings)
    S3MediaStore(None, settings)
    if settings.object_store_endpoint_url:
        from urllib.parse import urlsplit
        value = settings.object_store_endpoint_url
        parsed = urlsplit(value)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or any(c.isspace() for c in value)):
            raise ValueError('Invalid storage endpoint configuration')


@router.get('/api/v1/health/ready', tags=['operations'])
def ready(request: Request):
    try:
        validate_runtime(request.app.state.settings)
        with get_connect(request)() as conn:
            conn.execute('SELECT 1')
        return {'status': 'ready'}
    except Exception:
        emit('readiness_unavailable', error_code='READINESS_UNAVAILABLE')
        return JSONResponse(status_code=503, content={'status': 'not_ready'})


@router.get('/metrics', tags=['operations'], openapi_extra={'x-required-scopes': ['metrics:read']})
def metrics(request: Request,
            principal: Annotated[ServicePrincipal, Security(require_service, scopes=['metrics:read'])]):
    telemetry = request.app.state.metrics
    try:
        connect = get_connect(request)
    except Exception:
        def connect():
            raise RuntimeError('Database unavailable')
    best_effort(telemetry.snapshot, connect)
    try:
        content = telemetry.render()
    except Exception:
        # Collector failure is independent from application availability.
        content = b'# Metrics temporarily unavailable\n'
    return Response(content=content, media_type=CONTENT_TYPE_LATEST)
