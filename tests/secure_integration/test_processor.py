from contextlib import contextmanager
import json
from pathlib import Path

import cv2
import numpy as np

import pytest

from ugsl_ai_coach.comparison.engine import compare_movement, compare_precomputed_movement
from ugsl_ai_coach.comparison.models import ComparisonRequest, HandCorrespondence
from ugsl_ai_coach.comparison.policy import ComparisonPolicy
from ugsl_ai_coach.cv.pipeline import extract_video
from ugsl_ai_coach.infrastructure.object_store.s3 import S3MediaStore
from ugsl_ai_coach.integration.handoff.models import AnalysisWorkItem
from ugsl_ai_coach.media.ports import (CorruptReferenceProfile, IncompatibleReferenceProfile, UnsupportedLearnerVideo,
    ObjectStoreUnavailable, LearnerMediaNotFound, ReferenceProfileNotFound)
from ugsl_ai_coach.media.profiles import build_reference_profile, serialize_profile, load_profile
from ugsl_ai_coach.worker.processor import ProductionAnalysisProcessor
from ugsl_ai_coach.worker.runtime import AnalysisWorker
from .conftest import FakeS3, SyntheticExtractor


@pytest.fixture
def composition(settings, synthetic_video, submission):
    extraction = extract_video(synthetic_video, SyntheticExtractor())
    correspondence = HandCorrespondence(reference_label="Left", learner_label="Left", body_region="LEFT_HAND",
                                         establishment_basis="Explicit synthetic expert configuration")
    profile = build_reference_profile(extraction, profile_id=submission.reference_profile_ref,
                                      reference_id="synthetic-expert", extractor_identity="synthetic-model-v1",
                                      correspondence=correspondence)
    class Objects:
        video = FakeS3(synthetic_video.read_bytes())
        reference = FakeS3(serialize_profile(profile), "application/json")
        def selected(self, key):
            return self.video if key.startswith("learner-videos/") else self.reference
        def head_object(self, **kwargs):
            return self.selected(kwargs["Key"]).head_object(**kwargs)
        def get_object(self, **kwargs):
            return self.selected(kwargs["Key"]).get_object(**kwargs)
    objects = Objects()
    store = S3MediaStore(objects, settings)
    paths = []
    original = store.learner_video
    @contextmanager
    def recording(reference):
        with original(reference) as path:
            paths.append(path)
            yield path
    store.learner_video = recording
    extractor = SyntheticExtractor()
    processor = ProductionAnalysisProcessor(store, lambda: extractor, extractor_identity="synthetic-model-v1", settings=settings)
    work = AnalysisWorkItem(analysis_id="AN-000001", attempt_id=submission.attempt_id,
                           learner_video_ref=store.pin_learner(submission.learner_video_ref),
                           reference_profile_ref=store.pin_reference(submission.reference_profile_ref),
                           state="CLAIMED", delivery_count=1, claimed_at_ms=100, lease_expires_at_ms=1100)
    objects.video.calls.clear()
    objects.reference.calls.clear()
    return processor, work, profile, objects, paths, extractor, extraction


def test_real_m3_m4_composition_and_cleanup(composition, monkeypatch):
    processor, work, profile, objects, paths, extractor, extraction = composition
    import ugsl_ai_coach.worker.processor as module
    calls = []
    original_m3, original_m4 = module.extract_video, module.compare_precomputed_movement
    def m3(*args):
        calls.append("M3")
        return original_m3(*args)
    def m4(*args):
        calls.append("M4")
        return original_m4(*args)
    monkeypatch.setattr(module, "extract_video", m3)
    monkeypatch.setattr(module, "compare_precomputed_movement", m4)
    result = processor.process(work)
    assert calls == ["M3", "M4"]  # reference is precomputed, not extracted again
    assert result.status == "COMPLETED" and result.overall_score == 1.0
    assert result.findings[0].skill == "MOVEMENT" and result.analysis_id == work.analysis_id
    assert paths and all(not path.exists() and not path.parent.exists() for path in paths)
    assert extractor.closed
    assert objects.video.bodies[0].closed and objects.reference.bodies[0].closed


def test_precomputed_entry_point_identical_to_existing_m4(composition):
    _, work, profile, _, _, _, extraction = composition
    request = ComparisonRequest(analysis_id=work.analysis_id, attempt_id=work.attempt_id,
                                reference_id=profile.reference_id, correspondence=profile.correspondence)
    expected = compare_movement(extraction, extraction, request, profile.policy)
    actual = compare_precomputed_movement(profile.trajectory, profile.context, extraction, request, profile.policy)
    assert actual == expected


def test_insufficient_learner_evidence_abstains(composition):
    processor, work, _, _, paths, extractor, _ = composition
    extractor.empty = True
    result = processor.process(work)
    assert result.status == "UNANALYZABLE" and result.overall_score is None
    assert extractor.closed and all(not p.exists() for p in paths)


def test_processor_exception_cleans_media_and_extractor(composition):
    processor, work, _, _, paths, extractor, _ = composition
    extractor.error = True
    with pytest.raises(RuntimeError):
        processor.process(work)
    assert extractor.closed and all(not p.exists() and not p.parent.exists() for p in paths)


def test_unsupported_decoding_is_not_analysis_failed(composition):
    processor, work, _, objects, paths, extractor, _ = composition
    objects.video.data = b"synthetic not a video"
    with pytest.raises(UnsupportedLearnerVideo):
        processor.process(work)
    assert extractor.closed and all(not p.exists() for p in paths)


@pytest.mark.parametrize("value,expected", [(b"not json", CorruptReferenceProfile),
    (b'{}', CorruptReferenceProfile), (b'{"schema_version":"v999"}', IncompatibleReferenceProfile),
    (b'{"schema_version":"x","schema_version":"y"}', CorruptReferenceProfile),
    (b'{"bad":NaN}', CorruptReferenceProfile)])
def test_corrupt_or_incompatible_reference_fail_before_video(composition, value, expected):
    processor, work, _, objects, paths, _, _ = composition
    objects.reference.data = value
    with pytest.raises(expected):
        processor.process(work)
    assert objects.video.calls == [] and paths == []


@pytest.mark.parametrize("field,value", [("schema_version", "v999"), ("extraction_version", "m3-v999"),
    ("extractor_identity", "other-model"), ("profile_id", "reference-profiles/other.json")])
def test_reference_compatibility(composition, field, value):
    processor, work, profile, objects, paths, _, _ = composition
    data = profile.model_dump(mode="json")
    data[field] = value
    objects.reference.data = json.dumps(data).encode()
    with pytest.raises(IncompatibleReferenceProfile):
        processor.process(work)
    assert paths == []


@pytest.mark.parametrize("field", ["sampling", "normalization_config"])
def test_reference_context_incompatible(composition, field):
    processor, work, profile, objects, paths, _, _ = composition
    data = profile.model_dump(mode="json")
    if field == "sampling":
        data["context"][field]["target_fps"] = 30.0
    else:
        data["context"][field]["minimum_anchor_visibility"] = 0.8
    objects.reference.data = json.dumps(data).encode()
    with pytest.raises(IncompatibleReferenceProfile):
        processor.process(work)
    assert paths == []


def test_reference_policy_mismatch(composition):
    processor, work, profile, objects, paths, _, _ = composition
    data = profile.model_dump(mode="json")
    data["policy"]["minimum_handedness_confidence"] = 0.75
    objects.reference.data = json.dumps(data).encode()
    with pytest.raises(IncompatibleReferenceProfile):
        processor.process(work)


def test_storage_failure_does_not_fabricate_m2_failed(composition, settings, caplog):
    processor, work, _, objects, _, _, _ = composition
    from botocore.exceptions import EndpointConnectionError
    objects.reference.error = EndpointConnectionError(endpoint_url="https://synthetic-sensitive-host")
    class Persistence:
        finishes = []
        def claim_next(self, **kwargs):
            from ugsl_ai_coach.integration.models import AnalysisJob
            from ugsl_ai_coach.integration.handoff.models import JobWorkPair
            return JobWorkPair(job=AnalysisJob(analysis_id=work.analysis_id, attempt_id=work.attempt_id,
                state="PROCESSING", learner_video_ref=work.learner_video_ref, reference_profile_ref=work.reference_profile_ref,
                idempotency_key="synthetic"), work=work)
        def begin(self, expected):
            return self.claim_next()
        def finish(self, expected, result):
            self.finishes.append(result)
    persistence = Persistence()
    assert not AnalysisWorker(persistence, processor, settings).run_once()
    assert persistence.finishes == []
    assert "synthetic-sensitive-host" not in caplog.text
    assert "reference-profiles" not in caplog.text


def test_profile_roundtrip_minimal_no_raw_landmarks(composition):
    _, _, profile, _, _, _, _ = composition
    encoded = serialize_profile(profile)
    assert load_profile(encoded, profile_id=profile.profile_id, extractor_identity=profile.extractor_identity,
                        policy=profile.policy) == profile
    for forbidden in (b'"frames"', b'"raw"', b'"landmarks"', b'"bgr"', b'"z"'):
        assert forbidden not in encoded


@pytest.mark.parametrize("object_name,expected", [("video", LearnerMediaNotFound), ("reference", ReferenceProfileNotFound)])
def test_processor_missing_objects_distinguished(composition, object_name, expected):
    from botocore.exceptions import ClientError
    processor, work, _, objects, paths, _, _ = composition
    getattr(objects, object_name).error = ClientError({"Error": {"Code": "NoSuchKey", "Message": "synthetic-private-key"}}, "HeadObject")
    with pytest.raises(expected) as caught:
        processor.process(work)
    assert "synthetic-private-key" not in str(caught.value) and paths == []


def test_mp4_container_supported_by_real_m3_decoder(composition, tmp_path):
    processor, work, _, objects, paths, _, _ = composition
    video = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    assert writer.isOpened(), "The advertised MP4 container must be usable with the installed decoder"
    try:
        for index in range(12):
            writer.write(np.full((48, 64, 3), index * 5, dtype=np.uint8))
    finally:
        writer.release()
    objects.video.data, objects.video.content_type = video.read_bytes(), "video/mp4"
    work = work.model_copy(update={"learner_video_ref": processor.media.pin_learner("learner-videos/synthetic.mp4")})
    assert processor.process(work).status == "COMPLETED"
    assert paths[0].suffix == ".mp4" and not paths[0].exists()


@pytest.mark.integration
def test_production_factory_and_actual_local_mediapipe(composition, settings, synthetic_video, monkeypatch):
    models = Path(__file__).resolve().parents[2] / ".models"
    hand, pose = models / "hand_landmarker.task", models / "pose_landmarker_lite.task"
    if not hand.is_file() or not pose.is_file():
        pytest.skip("Optional integration smoke requires documented local .models assets")
    from ugsl_ai_coach.worker.processor import create_production_processor
    import ugsl_ai_coach.infrastructure.object_store.s3 as module
    prior, work, profile, objects, paths, _, _ = composition
    monkeypatch.setattr(module, "create_s3_media_store", lambda configured: prior.media)
    configured = settings.model_copy(update={"hand_model_path": str(hand), "pose_model_path": str(pose)})
    processor = create_production_processor(configured)
    # Both source profile and learner observation use the actual same model assets.
    with processor.extractor_factory() as extractor:
        reference = extract_video(synthetic_video, extractor)
    actual_profile = build_reference_profile(reference, profile_id=profile.profile_id, reference_id=profile.reference_id,
        extractor_identity=processor.extractor_identity, correspondence=profile.correspondence)
    objects.reference.data = serialize_profile(actual_profile)
    result = processor.process(work)
    assert result.status == "UNANALYZABLE" and result.overall_score is None
    assert all(not path.exists() for path in paths)
