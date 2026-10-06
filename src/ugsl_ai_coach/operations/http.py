"""Generated request correlation and safe route-template telemetry."""

from time import perf_counter
from uuid import uuid4

from ugsl_ai_coach.operations.events import emit, request_id
from ugsl_ai_coach.operations.metrics import best_effort


def install_operations(app):
    @app.middleware('http')
    async def correlate(request, call_next):
        rid = uuid4().hex
        token = request_id.set(rid)
        started = perf_counter()
        method = request.method if request.method in ('GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS') else 'OTHER'
        emit('request_started', method=method)
        try:
            response = await call_next(request)
            route = getattr(request.scope.get('route'), 'path', 'unmatched')
            # FastAPI's lazy included routers may retain their local template.
            if route in ('/health', '/contracts/analysis', '/analyses', '/analyses/{analysis_id}',
                         '/analyses/{analysis_id}/feedback'):
                route = '/api/v1' + route
            duration = perf_counter() - started
            response.headers['X-Request-ID'] = rid
            emit('request_completed', method=method, route=route, status=response.status_code, duration_seconds=duration)
            best_effort(app.state.metrics.record_http, route, method, response.status_code, duration)
            return response
        finally:
            request_id.reset(token)
