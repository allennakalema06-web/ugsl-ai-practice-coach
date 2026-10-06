"""Small JSON logging configuration using the standard library."""

import json
import logging
from logging.config import dictConfig


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in ("event", "analysis_id", "error_type", "request_id", "route", "method", "status",
                      "worker_type", "error_code", "duration_seconds", "outcome"):
            if hasattr(record, field):
                entry[field] = getattr(record, field)
        # Upstream exception text/tracebacks can contain credentials or evidence.
        return json.dumps(entry)


def configure_logging(level: str) -> None:
    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"json": {"()": JsonFormatter}},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "ugsl_ai_coach": {
                "handlers": ["console"], "level": level, "propagate": False
            }
        },
    })
