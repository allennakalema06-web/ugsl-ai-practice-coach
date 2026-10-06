"""Durable deterministic M5 delivery; production orchestration, no external LLM."""

from threading import Event
from typing import Protocol

from ugsl_ai_coach.coaching.engine import generate_feedback
from ugsl_ai_coach.coaching.providers.base import CoachingProvider
from ugsl_ai_coach.domain.analysis import StructuredAnalysisResult
from ugsl_ai_coach.integration.coaching_work import CoachingWork
from ugsl_ai_coach.integration.models import CoachingRecord
from ugsl_ai_coach.operations.events import emit
from ugsl_ai_coach.operations.metrics import best_effort, worker_metrics


class CoachingPersistence(Protocol):
    def claim_next(self, *, lease_duration_ms: int) -> CoachingWork | None: ...
    def existing(self, expected: CoachingWork) -> CoachingRecord | None: ...
    def load_analysis(self, expected: CoachingWork) -> StructuredAnalysisResult: ...
    def complete(self, expected: CoachingWork, record: CoachingRecord) -> CoachingRecord: ...
    def retry(self, expected: CoachingWork, *, base_seconds: int, max_seconds: int): ...


class CoachingProcessor:
    def __init__(self, provider: CoachingProvider | None = None):
        self.provider = provider

    def process(self, work: CoachingWork, analysis: StructuredAnalysisResult) -> CoachingRecord:
        work = CoachingWork.model_validate(work.model_dump(mode='python'))
        if analysis.analysis_id != work.analysis_id or analysis.attempt_id != work.attempt_id:
            raise ValueError('Coaching evidence must match delivery identity')
        feedback = generate_feedback(analysis, feedback_id=work.feedback_id, provider=self.provider)
        return CoachingRecord(analysis_id=work.analysis_id, attempt_id=work.attempt_id, feedback=feedback)


class CoachingWorker:
    def __init__(self, persistence: CoachingPersistence, processor: CoachingProcessor, settings, metrics=None):
        self.persistence, self.processor = persistence, processor
        self.settings = settings
        self.metrics = metrics if metrics is not None else worker_metrics
        self.stop = Event()

    def run_once(self):
        work = None
        try:
            work = self.persistence.claim_next(lease_duration_ms=self.settings.coaching_lease_seconds * 1000)
            if work is None:
                return False
            work = CoachingWork.model_validate(work.model_dump(mode='python'))
            if work.state != 'CLAIMED':
                raise ValueError('Worker requires coaching claim')
            emit('coaching_claimed', worker_type='coaching', analysis_id=work.analysis_id)
            if self.stop.is_set():
                return False
            stored = self.persistence.existing(work)
            if stored is None:
                analysis = self.persistence.load_analysis(work)
                record = self.processor.process(work, analysis)
                self.persistence.complete(work, record)
            best_effort(self.metrics.outcome, 'coaching', 'success')
            emit('coaching_completed', worker_type='coaching', analysis_id=work.analysis_id)
            return True
        except Exception:
            emit('coaching_processing_failed', worker_type='coaching', error_code='COACHING_PROCESSING_ERROR')
            if work is not None:
                try:
                    # Rereads committed truth: a lost completion ack must not reopen work.
                    if self.persistence.existing(work) is not None:
                        best_effort(self.metrics.outcome, 'coaching', 'success')
                        return True
                    self.persistence.retry(work, base_seconds=self.settings.coaching_retry_base_seconds,
                                           max_seconds=self.settings.coaching_retry_max_seconds)
                    best_effort(self.metrics.outcome, 'coaching', 'retry')
                    emit('coaching_retry_scheduled', worker_type='coaching', analysis_id=work.analysis_id)
                except Exception:
                    emit('coaching_recovery_deferred', worker_type='coaching', error_code='RECOVERY_UNAVAILABLE')
            best_effort(self.metrics.outcome, 'coaching', 'failure')
            return False
        finally:
            # Shared DB metrics run after delivery/retry, so a slow metric sink
            # cannot spend the active processing lease.
            if work is not None and isinstance(work, CoachingWork):
                best_effort(self.metrics.claim, 'coaching', work.delivery_count)

    def request_shutdown(self):
        self.stop.set()

    def run(self):
        emit('worker_started', worker_type='coaching')
        try:
            while not self.stop.is_set():
                self.run_once()
                self.stop.wait(self.settings.coaching_poll_seconds)
        finally:
            emit('worker_stopped', worker_type='coaching')


def main():
    from ugsl_ai_coach.core.config import Settings
    from ugsl_ai_coach.core.logging import configure_logging
    from ugsl_ai_coach.infrastructure.postgres.connection import connection_factory
    from ugsl_ai_coach.infrastructure.postgres.coaching_work import PostgresCoachingWork
    from ugsl_ai_coach.infrastructure.postgres.telemetry import PostgresWorkerMetrics
    from ugsl_ai_coach.worker.runtime import shutdown_signals
    configure_logging('INFO')
    try:
        settings = Settings()
        connect = connection_factory(settings)
        worker = CoachingWorker(PostgresCoachingWork(connect), CoachingProcessor(), settings, PostgresWorkerMetrics(connect))
        with shutdown_signals(worker):
            worker.run()
    except Exception:
        emit('worker_startup_failed', worker_type='coaching', error_code='STARTUP_UNAVAILABLE')
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
