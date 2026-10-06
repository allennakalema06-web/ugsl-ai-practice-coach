"""Thin HTTP application coordinator over unchanged durable integration services."""

from ugsl_ai_coach.integration.handoff.service import AnalysisHandoffService
from ugsl_ai_coach.integration.models import AnalysisSubmission, CoachingRecord
from ugsl_ai_coach.integration.ports import CoachingRecordRepository
from ugsl_ai_coach.media.references import MediaReferencePolicy
from ugsl_ai_coach.coaching.grounding import ground_analysis
from ugsl_ai_coach.coaching.validation import validate_feedback
from ugsl_ai_coach.integration.errors import AdapterContractError
from collections.abc import Callable
from contextlib import AbstractContextManager
from ugsl_ai_coach.media.ports import MediaReferenceResolver
from ugsl_ai_coach.media.pinned import parse_pinned


class AnalysisApiService:
    def __init__(self, handoff: AnalysisHandoffService, coaching: CoachingRecordRepository,
                 references: MediaReferencePolicy,
                 resolver_factory: Callable[[], AbstractContextManager[MediaReferenceResolver]]):
        self.handoff, self.coaching, self.references = handoff, coaching, references
        self.resolver_factory = resolver_factory

    def submit(self, submission: AnalysisSubmission):
        self.references.learner(submission.learner_video_ref)
        self.references.reference(submission.reference_profile_ref)
        with self.resolver_factory() as resolver:
            learner = resolver.pin_learner(submission.learner_video_ref)
            reference = resolver.pin_reference(submission.reference_profile_ref)
        learner_pin = parse_pinned(learner, "learner", self.references)
        reference_pin = parse_pinned(reference, "reference", self.references)
        if learner_pin.key != submission.learner_video_ref or reference_pin.key != submission.reference_profile_ref:
            raise AdapterContractError("Resolved object identities must match the submitted keys")
        canonical = AnalysisSubmission(attempt_id=submission.attempt_id, idempotency_key=submission.idempotency_key,
                                       learner_video_ref=learner, reference_profile_ref=reference)
        return self.handoff.submit(canonical)

    def get(self, analysis_id):
        return self.handoff.get(analysis_id).job

    def feedback(self, analysis_id):
        job = self.get(analysis_id)  # unknown analysis is distinct from pending feedback
        record = self.coaching.get(analysis_id)
        if record is not None:
            record = CoachingRecord.model_validate(record.model_dump(mode="python"))
            if not job.state.is_terminal or record.analysis_id != job.analysis_id or record.attempt_id != job.attempt_id:
                raise AdapterContractError("Feedback does not match the requested analysis")
            validate_feedback(record.feedback, ground_analysis(job.structured_analysis, record.feedback.feedback_id),
                              provider_type=record.feedback.provider_metadata.provider_type)
        return job, record


def get_api_service(request):
    injected = getattr(request.app.state, "analysis_api_service", None)
    if injected is not None:
        return injected
    from ugsl_ai_coach.infrastructure.postgres.connection import connection_factory
    from ugsl_ai_coach.infrastructure.postgres.persistence import PostgresAnalysisPersistence, PostgresAnalysisIdFactory
    from ugsl_ai_coach.infrastructure.postgres.coaching import PostgresCoachingRepository
    settings = request.app.state.settings
    connect = connection_factory(settings)
    def resolver():
        from ugsl_ai_coach.infrastructure.object_store.s3 import create_s3_media_store
        return create_s3_media_store(settings)
    return AnalysisApiService(AnalysisHandoffService(PostgresAnalysisPersistence(connect), PostgresAnalysisIdFactory()),
                              PostgresCoachingRepository(connect), MediaReferencePolicy(settings), resolver)
