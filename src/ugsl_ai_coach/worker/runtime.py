"""Bounded synchronous worker with DB-authoritative lease ownership."""

import logging
import signal
from contextlib import contextmanager
from threading import Event, current_thread, main_thread
from typing import Protocol

from ugsl_ai_coach.core.config import Settings
from ugsl_ai_coach.domain.analysis import StructuredAnalysisResult
from ugsl_ai_coach.integration.handoff.models import AnalysisWorkItem, JobWorkPair
from ugsl_ai_coach.integration.errors import AdapterContractError
from ugsl_ai_coach.integration.models import JobState

logger = logging.getLogger(__name__)


class AnalysisProcessor(Protocol):
    def process(self, work: AnalysisWorkItem) -> StructuredAnalysisResult: ...


class WorkerPersistence(Protocol):
    """DB-clock calling view of existing M6B operations; no new data contracts."""
    def claim_next(self, *, lease_duration_ms: int) -> JobWorkPair | None: ...
    def begin(self, expected: AnalysisWorkItem) -> JobWorkPair: ...
    def finish(self, expected: AnalysisWorkItem, result: StructuredAnalysisResult) -> JobWorkPair: ...


class AnalysisWorker:
    def __init__(self, persistence: WorkerPersistence, processor: AnalysisProcessor, settings: Settings):
        self.persistence = persistence
        self.processor = processor
        self.lease_duration_ms = settings.worker_lease_seconds * 1000
        self.poll_seconds = settings.worker_poll_seconds
        self.stop = Event()

    def run_once(self) -> bool:
        """Return whether a job completed; exceptions leave recoverable lease state."""
        analysis_id = None
        try:
            pair = self.persistence.claim_next(lease_duration_ms=self.lease_duration_ms)
            if pair is None:
                return False
            pair = JobWorkPair.model_validate(pair.model_dump(mode="python"))
            analysis_id = pair.job.analysis_id
            if pair.work.state != "CLAIMED":
                raise AdapterContractError("Worker requires a claimed work item")
            if self.stop.is_set():
                return False  # claim will expire; do not start new processing after shutdown
            active = self.persistence.begin(pair.work)
            active = JobWorkPair.model_validate(active.model_dump(mode="python"))
            if active.work != pair.work or active.job.submission != pair.job.submission or active.job.state != JobState.PROCESSING:
                raise AdapterContractError("Begin must preserve the claim and enter processing")
            result = self.processor.process(active.work)
            if not isinstance(result, StructuredAnalysisResult):
                raise TypeError("Processor must return a typed M2 result")
            result = StructuredAnalysisResult.model_validate(result.model_dump(mode="python"))
            if result.analysis_id != active.job.analysis_id or result.attempt_id != active.job.attempt_id:
                raise AdapterContractError("Processor result must match the active analysis")
            terminal = self.persistence.finish(active.work, result)
            terminal = JobWorkPair.model_validate(terminal.model_dump(mode="python"))
            if (terminal.job.structured_analysis != result or terminal.job.submission != active.job.submission
                    or terminal.work.delivery_count != active.work.delivery_count):
                raise AdapterContractError("Finish must preserve identity, generation and supplied result")
            logger.info("Analysis work completed", extra={"event": "analysis_completed", "analysis_id": analysis_id})
            return True
        except Exception as error:
            # Neither exception text nor traceback: either may expose DSNs/media
            # refs. Error class + validated ID permit operational correlation.
            logger.error("Worker iteration failed", extra={"event": "worker_iteration_failed",
                         "analysis_id": analysis_id, "error_type": type(error).__name__})
            return False

    def run(self) -> None:
        logger.info("worker_started")
        try:
            while not self.stop.is_set():
                self.run_once()
                # Wait after successful work, no work, AND failures: no busy loop.
                self.stop.wait(self.poll_seconds)
        finally:
            logger.info("worker_stopped")

    def request_shutdown(self) -> None:
        self.stop.set()


@contextmanager
def shutdown_signals(worker: AnalysisWorker):
    if current_thread() is not main_thread():
        raise RuntimeError("Signal handling must be installed on the main thread")
    previous = {}
    def shutdown(signum, frame):
        worker.request_shutdown()
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, shutdown)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run_postgres_worker(processor: AnalysisProcessor, settings: Settings | None = None) -> None:
    """Explicit composition entry point; callers supply a real processor/resolver.

    No fake CLI processor, auto-migration, HTTP route, or storage retrieval.
    """
    from ugsl_ai_coach.core.logging import configure_logging
    from ugsl_ai_coach.infrastructure.postgres.connection import connection_factory
    from ugsl_ai_coach.infrastructure.postgres.persistence import PostgresAnalysisPersistence
    settings = settings if settings is not None else Settings()
    connect = connection_factory(settings)  # fail missing configuration before starting
    configure_logging(settings.log_level)
    worker = AnalysisWorker(PostgresAnalysisPersistence(connect), processor, settings)
    with shutdown_signals(worker):
        worker.run()
