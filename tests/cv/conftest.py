import cv2
import numpy as np
import pytest


@pytest.fixture
def make_video(tmp_path):
    def create(fps=30, count=12):
        path = tmp_path / f"synthetic-{fps}-{count}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
        assert writer.isOpened(), "Synthetic video codec must be available"
        try:
            for index in range(count):
                writer.write(np.full((48, 64, 3), index * 5 % 256, dtype=np.uint8))
        finally:
            writer.release()
        return path
    return create
