"""Safe structured operational events with failure-isolated sinks."""

from contextvars import ContextVar
import logging

request_id = ContextVar('ugsl_request_id', default=None)
logger = logging.getLogger('ugsl_ai_coach.operations')
FIELDS = {'request_id', 'route', 'method', 'status', 'worker_type', 'analysis_id',
          'error_code', 'duration_seconds', 'outcome'}


def emit(event, **fields):
    try:
        safe = {key: value for key, value in fields.items() if key in FIELDS}
        if request_id.get() is not None:
            safe['request_id'] = request_id.get()
        write = logger.error if event.endswith(('failed', 'unavailable', 'deferred')) or fields.get('status', 0) >= 500 else logger.info
        write(event, extra={'event': event, **safe})
    except Exception:
        pass  # Logging is never an analysis/coaching state transition.
