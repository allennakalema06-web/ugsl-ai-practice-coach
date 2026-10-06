"""Private media -> unchanged M3 extraction -> shared M4 comparison -> M2."""

from hashlib import sha256
from pathlib import Path
from collections.abc import Callable
from contextlib import AbstractContextManager

from ugsl_ai_coach.comparison.engine import compare_precomputed_movement
from ugsl_ai_coach.comparison.models import ComparisonRequest
from ugsl_ai_coach.comparison.policy import ComparisonPolicy
from ugsl_ai_coach.core.config import Settings
from ugsl_ai_coach.cv.landmarks import MediaPipeExtractor, LandmarkExtractor
from ugsl_ai_coach.domain.analysis import StructuredAnalysisResult
from ugsl_ai_coach.cv.models import NormalizationConfig, SamplingConfig
from ugsl_ai_coach.cv.pipeline import extract_video
from ugsl_ai_coach.cv.video import VideoError
from ugsl_ai_coach.integration.handoff.models import AnalysisWorkItem
from ugsl_ai_coach.media.ports import MediaStore, IncompatibleReferenceProfile, UnsupportedLearnerVideo
from ugsl_ai_coach.media.profiles import load_profile
from ugsl_ai_coach.media.references import MediaConfigurationError, MediaReferencePolicy
from ugsl_ai_coach.media.pinned import parse_pinned


class ProductionAnalysisProcessor:
    def __init__(self, media: MediaStore, extractor_factory: Callable[[], AbstractContextManager[LandmarkExtractor]], *, extractor_identity: str,
                 settings: Settings, policy: ComparisonPolicy | None = None):
        self.media, self.extractor_factory = media, extractor_factory
        self.extractor_identity = extractor_identity
        self.references = MediaReferencePolicy(settings)
        self.policy = policy if policy is not None else ComparisonPolicy()
        self.sampling, self.normalization = SamplingConfig(), NormalizationConfig()

    def process(self, work: AnalysisWorkItem) -> StructuredAnalysisResult:
        work = AnalysisWorkItem.model_validate(work.model_dump(mode="python"))
        parse_pinned(work.learner_video_ref, "learner", self.references)
        reference_pin = parse_pinned(work.reference_profile_ref, "reference", self.references)
        profile = load_profile(self.media.reference_profile(work.reference_profile_ref),
                               profile_id=reference_pin.key, extractor_identity=self.extractor_identity,
                               policy=self.policy)
        if profile.context.sampling != self.sampling or profile.context.normalization_config != self.normalization:
            raise IncompatibleReferenceProfile("Reference normalization or sampling is incompatible")
        with self.media.learner_video(work.learner_video_ref) as path:
            with self.extractor_factory() as extractor:
                try:
                    learner = extract_video(path, extractor, self.sampling, self.normalization)
                except VideoError:
                    raise UnsupportedLearnerVideo("Learner video cannot be decoded by the supported pipeline") from None
            request = ComparisonRequest(analysis_id=work.analysis_id, attempt_id=work.attempt_id,
                                        reference_id=profile.reference_id, correspondence=profile.correspondence)
            return compare_precomputed_movement(profile.trajectory, profile.context, learner, request, self.policy).analysis


def create_production_processor(settings: Settings) -> ProductionAnalysisProcessor:
    if not settings.hand_model_path or not settings.pose_model_path:
        raise MediaConfigurationError("Configured local hand and pose model assets are required")
    hand, pose = Path(settings.hand_model_path), Path(settings.pose_model_path)
    if not hand.is_file() or not pose.is_file():
        raise MediaConfigurationError("Configured local model assets are unavailable")
    identity = "mediapipe-tasks-1.0.1:" + sha256(hand.read_bytes()).hexdigest() + ":" + sha256(pose.read_bytes()).hexdigest()
    from ugsl_ai_coach.infrastructure.object_store.s3 import create_s3_media_store
    return ProductionAnalysisProcessor(create_s3_media_store(settings), lambda: MediaPipeExtractor(hand, pose),
                                       extractor_identity=identity, settings=settings)


def main():
    from ugsl_ai_coach.worker.runtime import run_postgres_worker
    from ugsl_ai_coach.core.logging import configure_logging
    from ugsl_ai_coach.infrastructure.postgres.connection import connection_factory
    configure_logging("INFO")
    try:
        settings = Settings()
        connection_factory(settings)  # validate DB configuration before S3/model composition
        run_postgres_worker(create_production_processor(settings), settings)
    except Exception:
        from ugsl_ai_coach.operations.events import emit
        emit('worker_startup_failed', worker_type='analysis', error_code='STARTUP_UNAVAILABLE')
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
