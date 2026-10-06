from hashlib import sha256
from importlib.resources import files
import json
import logging
from pathlib import Path
import re
import subprocess
import sys

from ugsl_ai_coach.core.logging import JsonFormatter
from ugsl_ai_coach.operations.metrics import Metrics


def test_new_runtime_imports_do_not_connect_clean_or_load_cv():
    code = '''
import psycopg, boto3, socket
def reject(*args, **kwargs): raise AssertionError('Unexpected infrastructure I/O')
psycopg.connect = reject
boto3.client = reject
socket.create_connection = reject
from ugsl_ai_coach.media.temporary import TemporaryMedia
TemporaryMedia.prepare = reject
TemporaryMedia.cleanup_stale = reject
import ugsl_ai_coach.main
import ugsl_ai_coach.worker.coaching
import ugsl_ai_coach.worker.runtime
import ugsl_ai_coach.infrastructure.postgres.coaching_work
import ugsl_ai_coach.infrastructure.postgres.rate_limit
import ugsl_ai_coach.infrastructure.postgres.telemetry
assert 'mediapipe' not in __import__('sys').modules
'''
    completed = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr


def test_migration_order_and_001_checksum_unchanged():
    root = files('ugsl_ai_coach.infrastructure.postgres').joinpath('migrations')
    assert sorted(p.name for p in root.iterdir() if p.name.endswith('.sql')) == [
        '001_initial_integration.sql', '002_operational_hardening.sql']
    script = root.joinpath('001_initial_integration.sql').read_text(encoding='utf-8')
    assert sha256(script.encode()).hexdigest() == '666d9f1130021a87bb1524460ed77c51590d68e64dbc5d06f2d417af543f79f4'


def test_alerts_reference_actual_metrics_and_no_delivery_integration():
    text = (Path(__file__).parents[2] / 'ops/prometheus/alerts.yml').read_text(encoding='utf-8')
    names = set(re.findall(r'ugsl_[a-z_]+', '\n'.join(line for line in text.splitlines() if 'expr:' in line)))
    actual = {family.name for family in Metrics().registry.collect()}
    # Prometheus Counter families expose the _total sample suffix.
    actual |= {name + '_total' for name in actual}
    assert names <= actual
    assert text.count('      - alert:') == 7
    for forbidden in ('slack', 'pagerduty', 'email', 'webhook'): assert forbidden not in text.lower()


def test_json_formatter_omits_sensitive_exceptions_and_unapproved_fields():
    try:
        raise RuntimeError('sensitive-token media-key learner@example.invalid')
    except RuntimeError:
        record = logging.LogRecord('ugsl_ai_coach', logging.ERROR, __file__, 1, 'database_unavailable', (), sys.exc_info())
    record.event = 'database_unavailable'
    record.authorization = 'Bearer secret'
    record.body = {'learner': 'private'}
    text = JsonFormatter().format(record)
    assert json.loads(text)['event'] == 'database_unavailable'
    for forbidden in ('sensitive-token', 'media-key', 'learner@example', 'Bearer', 'private', 'traceback'):
        assert forbidden not in text
