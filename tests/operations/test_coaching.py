import json
import logging

import pytest
from pydantic import ValidationError

from ugsl_ai_coach.coaching.providers.deterministic import DeterministicCoachingProvider
from ugsl_ai_coach.integration.coaching_work import CoachingWork, retry_seconds
from ugsl_ai_coach.operations.metrics import Metrics
from ugsl_ai_coach.worker.coaching import CoachingProcessor, CoachingWorker
from ugsl_ai_coach.worker.runtime import shutdown_signals


class Store:
    def __init__(self, work, result):
        self.work, self.result, self.record = work, result, None
        self.retries, self.loads = [], 0
        self.ack_lost = False
    def claim_next(self, **kwargs): return self.work
    def existing(self, work): return self.record
    def load_analysis(self, work):
        self.loads += 1
        return self.result
    def complete(self, work, record):
        self.record = record
        if self.ack_lost:
            raise RuntimeError('secret acknowledgement failure')
        return record
    def retry(self, work, **kwargs): self.retries.append(kwargs)


@pytest.mark.parametrize('status', ['COMPLETED', 'UNANALYZABLE', 'FAILED'])
def test_deterministic_m5_composition(work, result, status):
    from ugsl_ai_coach.domain.analysis import StructuredAnalysisResult
    analysis = StructuredAnalysisResult.model_validate({**result.model_dump(), 'status': status,
        'overall_score': 0.5 if status == 'COMPLETED' else None, 'overall_confidence': None if status == 'FAILED' else 1.0})
    processor = CoachingProcessor()
    first = processor.process(work, analysis)
    assert first == processor.process(work, analysis)
    assert first.feedback.feedback_id == work.feedback_id
    assert first.feedback.overall_status == status


@pytest.mark.parametrize('invalid', [False, True])
def test_provider_failure_never_mutates_evidence(work, result, settings, invalid, caplog):
    class Provider:
        provider_type = 'deterministic'
        def generate(self, context):
            if invalid:
                return DeterministicCoachingProvider().generate(context).model_copy(update={'summary': 'Invented claim'})
            raise RuntimeError('secret-key learner@example.invalid')
    store, telemetry = Store(work, result), Metrics()
    before = result.model_dump_json()
    with caplog.at_level(logging.INFO, logger='ugsl_ai_coach.operations'):
        assert not CoachingWorker(store, CoachingProcessor(Provider()), settings, telemetry).run_once()
    assert result.model_dump_json() == before and store.record is None
    assert store.retries == [{'base_seconds': 5, 'max_seconds': 300}]
    text = telemetry.render().decode()
    assert 'outcome="retry",worker="coaching"} 1.0' in text
    assert 'coaching_retry_scheduled' in caplog.text
    assert 'secret-key' not in caplog.text and 'learner@example.invalid' not in caplog.text


def test_ack_loss_rereads_truth_without_retry_or_second_provider(work, result, settings):
    store = Store(work, result)
    store.ack_lost = True
    worker = CoachingWorker(store, CoachingProcessor(), settings)
    assert worker.run_once() and store.record is not None and store.retries == []
    assert worker.run_once() and store.loads == 1


def test_existing_feedback_skips_provider(work, result, settings):
    store = Store(work, result)
    store.record = CoachingProcessor().process(work, result)
    class Reject:
        def process(self, *args): raise AssertionError('Provider should not be called')
    assert CoachingWorker(store, Reject(), settings).run_once()
    assert store.loads == 0


def test_telemetry_and_log_sink_failures_do_not_fail_delivery(work, result, settings, monkeypatch):
    import ugsl_ai_coach.operations.events as events
    def fail(*args, **kwargs): raise RuntimeError('sink failure')
    monkeypatch.setattr(events.logger, 'info', fail)
    class BrokenMetrics:
        claim = outcome = fail
    store = Store(work, result)
    assert CoachingWorker(store, CoachingProcessor(), settings, BrokenMetrics()).run_once()
    assert store.record is not None


def test_shutdown_skips_provider(work, result, settings):
    store = Store(work, result)
    worker = CoachingWorker(store, CoachingProcessor(), settings)
    worker.request_shutdown()
    assert not worker.run_once() and store.loads == 0


def test_bad_job_does_not_kill_bounded_loop(work, result, settings):
    class Broken:
        def claim_next(self, **kwargs): raise RuntimeError('secret')
    worker = CoachingWorker(Broken(), CoachingProcessor(), settings)
    class Stop:
        waits = []
        def is_set(self): return len(self.waits) == 3
        def wait(self, seconds): self.waits.append(seconds)
    worker.stop = Stop()
    worker.run()
    assert worker.stop.waits == [1.0] * 3


@pytest.mark.parametrize('failures,expected', [(0, 5), (1, 10), (5, 160), (6, 300), (1000000, 300)])
def test_bounded_backoff(failures, expected):
    assert retry_seconds(failures, 5, 300) == expected


@pytest.mark.parametrize('change', [{'delivery_count': 0}, {'failure_count': -1}, {'claimed_at': None},
    {'state': 'FAILED'}, {'last_error_code': 'secret exception'}, {'state': 'PENDING'}])
def test_work_contract_rejects_invalid_operational_shapes(work, change):
    with pytest.raises(ValidationError):
        CoachingWork.model_validate({**work.model_dump(), **change})


@pytest.mark.parametrize('field,value', [('coaching_lease_seconds', 0), ('coaching_poll_seconds', float('inf')),
    ('coaching_retry_base_seconds', 0), ('coaching_retry_max_seconds', 1), ('submit_rate_limit_per_minute', 0),
    ('read_rate_limit_per_minute', 1000001), ('temp_media_max_age_seconds', 1)])
def test_configuration_bounds(field, value):
    from ugsl_ai_coach.core.config import Settings
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})
