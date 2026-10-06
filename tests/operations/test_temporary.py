import os
from pathlib import Path
from time import time

import pytest

from ugsl_ai_coach.media.temporary import TemporaryMedia, MARKER, exclusive


def residue(media, age, name='work-abcdefgh'):
    root = media.prepare()
    directory = root / name
    directory.mkdir()
    (directory / '.owner').write_bytes(MARKER)
    (directory / '.lock').write_bytes(b'0')
    (directory / 'input.mp4').write_bytes(b'synthetic')
    stamp = time() - age
    os.utime(directory, (stamp, stamp))
    return directory


@pytest.mark.parametrize('error', [False, True])
def test_normal_and_exception_cleanup(tmp_path, error):
    media = TemporaryMedia(tmp_path / 'owned')
    paths = []
    def run():
        with media.directory() as directory:
            paths.append(directory)
            (directory / 'input.avi').write_bytes(b'synthetic')
            if error: raise RuntimeError('synthetic')
    if error:
        with pytest.raises(RuntimeError): run()
    else: run()
    assert not paths[0].exists()
    assert (media.root / '.owner').is_file()


def test_only_stale_owned_residue_removed(tmp_path):
    media = TemporaryMedia(tmp_path / 'owned')
    stale = residue(media, 90000)
    fresh = residue(media, 0, 'work-freshdir')
    unrelated = media.root / 'arbitrary.txt'
    unrelated.write_bytes(b'keep')
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'input.mp4').write_bytes(b'keep')
    assert media.cleanup_stale(86400) == 1
    assert not stale.exists() and fresh.is_dir() and unrelated.read_bytes() == b'keep'
    assert (outside / 'input.mp4').read_bytes() == b'keep'


def test_old_but_actively_locked_directory_preserved(tmp_path):
    media = TemporaryMedia(tmp_path / 'owned')
    directory = residue(media, 90000)
    with exclusive(directory / '.lock'):
        assert media.cleanup_stale(86400) == 0
        assert (directory / 'input.mp4').is_file()
    assert media.cleanup_stale(86400) == 1


@pytest.mark.parametrize('damage', ['marker', 'extra', 'unmarked', 'escape'])
def test_unowned_or_unexpected_artifacts_refused(tmp_path, damage):
    media = TemporaryMedia(tmp_path / 'owned')
    directory = residue(media, 90000)
    if damage == 'marker': (directory / '.owner').write_bytes(b'other service')
    elif damage == 'extra': (directory / 'other.txt').write_bytes(b'keep')
    elif damage == 'unmarked': (directory / '.owner').unlink()
    else:
        with pytest.raises(ValueError): media.remove(tmp_path)
        return
    assert media.cleanup_stale(86400) == 0 and directory.is_dir()


def test_existing_unowned_root_is_not_adopted(tmp_path):
    root = tmp_path / 'arbitrary'
    root.mkdir()
    with pytest.raises(ValueError): TemporaryMedia(root).cleanup_stale(86400)
    assert not (root / '.owner').exists()


def test_symlink_artifacts_never_followed(tmp_path):
    media = TemporaryMedia(tmp_path / 'owned')
    directory = residue(media, 90000)
    target = tmp_path / 'keep.txt'
    target.write_bytes(b'keep')
    (directory / 'input.mp4').unlink()
    try:
        (directory / 'input.mp4').symlink_to(target)
    except OSError:
        # Windows without link privilege: exercise the same lstat safety boundary
        # as a reparse point with a narrowly scoped fake, without claiming OS links.
        from unittest.mock import patch
        import ugsl_ai_coach.media.temporary as module
        (directory / 'input.mp4').write_bytes(b'keep')
        original = module.unsafe
        with patch.object(module, 'unsafe', lambda p: p.name == 'input.mp4' or original(p)):
            assert media.cleanup_stale(86400) == 0
    else:
        assert media.cleanup_stale(86400) == 0
    assert target.read_bytes() == b'keep' and directory.is_dir()


def test_root_reparse_point_refused_without_touching_target(tmp_path, monkeypatch):
    import ugsl_ai_coach.media.temporary as module
    root = tmp_path / 'owned'
    root.mkdir()
    original = module.unsafe
    monkeypatch.setattr(module, 'unsafe', lambda p: p == root or original(p))
    with pytest.raises(ValueError): TemporaryMedia(root).prepare()
    assert root.is_dir()


def test_cleanup_has_no_object_store_dependency(tmp_path, monkeypatch):
    import boto3
    def reject(*args, **kwargs): raise AssertionError('No object operation permitted')
    monkeypatch.setattr(boto3, 'client', reject)
    media = TemporaryMedia(tmp_path / 'owned')
    residue(media, 90000)
    assert media.cleanup_stale(86400) == 1
