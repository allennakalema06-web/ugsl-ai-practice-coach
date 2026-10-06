import json
import logging
import signal
from threading import Event

import pytest

from ugsl_ai_coach.core.config import Settings
from ugsl_ai_coach.core.logging import JsonFormatter
from ugsl_ai_coach.integration.handoff.lifecycle import ClaimConflict, claim_work, complete_work
from ugsl_ai_coach.integration.handoff.models import JobWorkPair
from ugsl_ai_coach.integration.lifecycle import transition_job
from ugsl_ai_coach.integration.models import JobState
from ugsl_ai_coach.worker.runtime import AnalysisWorker, shutdown_signals


class Store:
    def __init__(self, pair):
        self.pair = pair
        self.finished = []
        self.failure = None
    def claim_next(self, *, lease_duration_ms):
        if self.pair is None:
            return None
        self.pair = JobWorkPair(job=self.pair.job, work=claim_work(
            self.pair.work, now_ms=100, lease_duration_ms=lease_duration_ms))
        return self.pair
    def begin(self, expected):
        if self.failure == "stale":
            raise ClaimConflict("stale")
        self.pair = JobWorkPair(job=transition_job(self.pair.job, JobState.PROCESSING), work=expected)
        return self.pair
    def finish(self, expected, result):
        if self.failure == "finish":
            raise RuntimeError("synthetic-sensitive-ref")
        self.finished.append(result)
        self.pair = JobWorkPair(job=transition_job(self.pair.job, JobState(result.status.value),
                               structured_analysis=result), work=complete_work(expected, now_ms=101))
        return self.pair


class Processor:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []
    def process(self, work):
        self.calls.append(work)
        if self.error:
            raise self.error
        return self.result


def worker(store, processor):
    return AnalysisWorker(store, processor, Settings(_env_file=None))


@pytest.mark.parametrize("status", ["COMPLETED", "UNANALYZABLE", "FAILED"])
def test_worker_persists_only_actual_result(make_pair, make_result, status):
    store, result = Store(make_pair()), make_result(status)
    processor = Processor(result)
    assert worker(store, processor).run_once()
    assert store.finished == [result]
    assert store.pair.job.structured_analysis == result
    assert store.pair.work.state == "COMPLETED"
    assert processor.calls[0].delivery_count == 1


def test_processor_exception_keeps_processing_and_no_failed_result(make_pair, caplog):
    store = Store(make_pair())
    processor = Processor(error=RuntimeError("synthetic-sensitive-ref"))
    with caplog.at_level(logging.ERROR):
        assert not worker(store, processor).run_once()
    assert store.pair.job.state == "PROCESSING"
    assert store.pair.work.state == "CLAIMED"
    assert store.pair.job.structured_analysis is None and store.finished == []
    failure = caplog.records[-1]
    assert failure.analysis_id == "AN-000001" and failure.error_code == "ANALYSIS_PROCESSING_ERROR"
    assert "synthetic-sensitive-ref" not in caplog.text
    assert "synthetic-video" not in caplog.text
    structured = json.loads(JsonFormatter().format(failure))
    assert structured["analysis_id"] == "AN-000001" and structured["error_code"] == "ANALYSIS_PROCESSING_ERROR"
    assert structured["event"] == "analysis_processing_failed"
    assert "synthetic-sensitive-ref" not in json.dumps(structured)


@pytest.mark.parametrize("failure", ["stale", "finish"])
def test_stale_and_persistence_exceptions_do_not_fabricate_result(make_pair, make_result, failure):
    store = Store(make_pair())
    store.failure = failure
    processor = Processor(make_result())
    assert not worker(store, processor).run_once()
    assert store.finished == [] and store.pair.job.structured_analysis is None
    assert len(processor.calls) == (0 if failure == "stale" else 1)


@pytest.mark.parametrize("invalid", [None, {}, "FAILED"])
def test_untyped_processor_output_is_not_persisted(make_pair, invalid):
    store = Store(make_pair())
    assert not worker(store, Processor(invalid)).run_once()
    assert store.finished == []


def test_unchecked_processor_output_rejected(make_pair, make_result):
    store = Store(make_pair())
    invalid = make_result().model_copy(update={"overall_score": 5.0})
    assert not worker(store, Processor(invalid)).run_once()
    assert store.finished == []


def test_processor_result_wrong_identity_rejected(make_pair, make_result):
    store = Store(make_pair())
    assert not worker(store, Processor(make_result(analysis_id="AN-000002"))).run_once()
    assert store.finished == []


def test_malformed_job_does_not_kill_loop(make_pair):
    malformed = make_pair().model_copy(update={"work": make_pair().work.model_copy(update={"delivery_count": -1})})
    class MalformedStore:
        calls = 0
        def claim_next(self, **kwargs):
            self.calls += 1
            return malformed if self.calls == 1 else None
    store = MalformedStore()
    runtime = worker(store, Processor())
    class Stop:
        waits = 0
        def is_set(self):
            return self.waits == 2
        def wait(self, seconds):
            self.waits += 1
    runtime.stop = Stop()
    runtime.run()
    assert store.calls == 2


def test_no_work_does_not_invoke_processor():
    processor = Processor()
    assert not worker(Store(None), processor).run_once()
    assert processor.calls == []


@pytest.mark.parametrize("exception", [False, True])
def test_loop_waits_and_recovers_after_each_iteration(exception, caplog):
    class PollStore:
        calls = 0
        def claim_next(self, **kwargs):
            self.calls += 1
            if exception and self.calls == 1:
                raise RuntimeError("secret")
            return None
    store = PollStore()
    runtime = worker(store, Processor())
    class Stop:
        waits = []
        def is_set(self):
            return len(self.waits) == 3
        def wait(self, seconds):
            self.waits.append(seconds)
    runtime.stop = Stop()
    runtime.run()
    assert store.calls == 3 and runtime.stop.waits == [1.0, 1.0, 1.0]
    if exception:
        assert any(getattr(record, "event", None) == "analysis_processing_failed" for record in caplog.records)
        assert "secret" not in caplog.text


def test_shutdown_skips_new_processing(make_pair):
    store = Store(make_pair())
    processor = Processor()
    runtime = worker(store, processor)
    runtime.request_shutdown()
    assert not runtime.run_once()
    assert processor.calls == []
    runtime.run()  # already stopped: no new claim


def test_shutdown_signal_restores_handlers(monkeypatch):
    handlers = {signal.SIGINT: "original-int", signal.SIGTERM: "original-term"}
    def install(signum, handler):
        previous = handlers[signum]
        handlers[signum] = handler
        return previous
    monkeypatch.setattr(signal, "signal", install)
    runtime = worker(Store(None), Processor())
    with shutdown_signals(runtime):
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        assert runtime.stop.is_set()
    assert handlers == {signal.SIGINT: "original-int", signal.SIGTERM: "original-term"}
