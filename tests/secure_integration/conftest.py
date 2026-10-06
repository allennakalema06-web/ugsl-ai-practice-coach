from contextlib import contextmanager
from io import BytesIO

import cv2
import numpy as np
import pytest

from ugsl_ai_coach.core.config import Settings
from ugsl_ai_coach.cv.models import Coordinates, HandObservation, Landmark, RawObservation
from ugsl_ai_coach.infrastructure.object_store.s3 import S3MediaStore
from ugsl_ai_coach.integration.models import AnalysisSubmission

SYNTHETIC_TOKEN = "m6d-test-token-0123456789ABCDEFGHIJKLMNOP"


@pytest.fixture
def settings():
    return Settings(_env_file=None, service_token=SYNTHETIC_TOKEN, object_store_bucket="synthetic-private-bucket")


@pytest.fixture
def submission():
    return AnalysisSubmission(attempt_id="ATT-000001", learner_video_ref="learner-videos/synthetic.avi",
                              reference_profile_ref="reference-profiles/synthetic.json", idempotency_key="synthetic-key")


class FakeS3:
    def __init__(self, data=b"synthetic-video", content_type="video/x-msvideo"):
        self.data, self.content_type = data, content_type
        self.calls, self.bodies = [], []
        self.head_override, self.get_override = {}, {}
        self.error = None
    def close(self):
        pass
    def head_object(self, **kwargs):
        self.calls.append(("head", kwargs))
        if self.error:
            raise self.error
        return {"ContentLength": len(self.data), "ContentType": self.content_type,
                "ETag": '"synthetic-etag"', **self.head_override}
    def get_object(self, **kwargs):
        self.calls.append(("get", kwargs))
        if self.error:
            raise self.error
        body = BytesIO(self.data)
        self.bodies.append(body)
        return {"Body": body, "ContentLength": len(self.data), "ContentType": self.content_type,
                "ETag": self.head_override.get("ETag", '"synthetic-etag"'),
                "VersionId": self.head_override.get("VersionId"), **self.get_override}


@pytest.fixture
def s3(settings):
    client = FakeS3()
    return client, S3MediaStore(client, settings)


class SyntheticExtractor:
    def __init__(self, *, empty=False, error=False):
        self.empty, self.error, self.closed = empty, error, False
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.closed = True
    def extract(self, frame):
        if self.error:
            raise RuntimeError("synthetic-sensitive-landmarker-error")
        if self.empty:
            return RawObservation()
        pose = tuple(Landmark(index=i, coordinates=Coordinates(x=x, y=0.5), visibility=1.0)
                     for i, x in [(11, 0.25), (12, 0.75)])
        hand = HandObservation(reported_handedness="Left", handedness_confidence=0.9,
            landmarks=tuple(Landmark(index=i, coordinates=Coordinates(x=0.3 + frame.index / 100, y=0.5))
                            for i in range(21)))
        return RawObservation(hands=(hand,), pose=pose)


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (64, 48))
    assert writer.isOpened()
    try:
        for index in range(12):
            writer.write(np.full((48, 64, 3), index * 5, dtype=np.uint8))
    finally:
        writer.release()
    return path
