"""Only flat, marked service-owned media directories are ever removed.

File locks protect active downloads even if older than the stale-age threshold.
No recursive deletion, symlink traversal, object deletion or import-time cleanup.
"""

from contextlib import contextmanager
import os
from pathlib import Path
import re
import stat
from tempfile import gettempdir, mkdtemp
from time import time

MARKER = b'ugsl-ai-owned-media-v1\n'


def unsafe(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def guard_path(path):
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            if unsafe(part):
                raise ValueError('Temporary media links/reparse points are refused')


@contextmanager
def exclusive(path):
    if unsafe(path) or not path.is_file():
        raise ValueError('Invalid media ownership lock')
    with path.open('r+b') as lock:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == 'nt':
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


class TemporaryMedia:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else Path(gettempdir()) / 'ugsl-ai-owned-media-v1'

    def prepare(self):
        guard_path(self.root)
        try:
            self.root.mkdir(mode=0o700)
        except FileExistsError:
            marker = self.root / '.owner'
            if not marker.is_file() or unsafe(marker) or marker.read_bytes() != MARKER:
                raise ValueError('Unowned temporary media root')
        else:
            (self.root / '.owner').write_bytes(MARKER)
        return self.root

    def owned_files(self, directory):
        guard_path(directory)
        if directory.parent != self.root or not re.fullmatch(r'work-[a-z0-9_]{8,}', directory.name):
            raise ValueError('Invalid temporary media directory')
        files = list(directory.iterdir())
        if any(p.name not in ('.owner', '.lock', 'input.avi', 'input.mp4') or unsafe(p) or not p.is_file() for p in files):
            raise ValueError('Unknown or linked media artifact')
        marker = directory / '.owner'
        if not marker.is_file() or marker.read_bytes() != MARKER:
            raise ValueError('Unowned media directory')
        return files

    def remove(self, directory):
        files = self.owned_files(directory)
        for item in files:
            item.unlink()
        directory.rmdir()

    @contextmanager
    def directory(self):
        root = self.prepare()
        directory = Path(mkdtemp(prefix='work-', dir=root))
        (directory / '.owner').write_bytes(MARKER)
        (directory / '.lock').write_bytes(b'0')
        try:
            with exclusive(directory / '.lock'):
                yield directory
        finally:
            self.remove(directory)

    def cleanup_stale(self, max_age_seconds, *, now=None):
        if type(max_age_seconds) is not int or not 3600 <= max_age_seconds <= 604800:
            raise ValueError('Invalid stale media age')
        root = self.prepare()
        cutoff = (time() if now is None else now) - max_age_seconds
        removed = 0
        for directory in root.iterdir():
            if not directory.name.startswith('work-'):
                continue
            try:
                self.owned_files(directory)
                if directory.stat().st_mtime >= cutoff:
                    continue
                with exclusive(directory / '.lock'):
                    self.owned_files(directory)
                    # Keep the lock file until it is released (Windows sharing).
                    for item in directory.iterdir():
                        if item.name != '.lock':
                            item.unlink()
                (directory / '.lock').unlink()
                directory.rmdir()
                removed += 1
            except (OSError, ValueError):
                continue  # Active/unowned/linked artifacts are never eligible.
        return removed
