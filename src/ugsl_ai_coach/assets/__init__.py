"""Reviewed M3 version-1 assets, packaged byte-for-byte; never downloaded at runtime."""

from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path


def model_paths() -> tuple[Path, Path]:
    root = files(__package__)
    manifest = json.loads(root.joinpath('manifest.json').read_text(encoding='utf-8'))
    paths = []
    for name in ('hand_landmarker.task', 'pose_landmarker_lite.task'):
        # Normal wheel installs are unpacked. Refuse a zip-only install instead
        # of relying on an ephemeral extraction after processor construction.
        path = Path(str(root.joinpath(name)))
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != manifest[name]['sha256']:
            raise ValueError('Packaged model asset missing or checksum mismatch')
        paths.append(path)
    return tuple(paths)


def extractor_identity() -> str:
    hand, pose = model_paths()
    return 'mediapipe-tasks-1.0.1:' + sha256(hand.read_bytes()).hexdigest() + ':' + sha256(pose.read_bytes()).hexdigest()
