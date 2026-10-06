import json
import logging

from ugsl_ai_coach.core.logging import JsonFormatter


def test_log_output_is_structured_json():
    record = logging.LogRecord("ugsl_ai_coach", logging.INFO, __file__, 1, "Service %s", ("starting",), None)
    assert json.loads(JsonFormatter().format(record)) == {
        "level": "INFO", "logger": "ugsl_ai_coach", "message": "Service starting"
    }
