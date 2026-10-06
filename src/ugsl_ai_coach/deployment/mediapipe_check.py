"""Credential-free, real packaged-model initialization diagnostic; not a logger.

Each landmarker runs in a separate child with a minimal environment. Native
stdout/stderr is captured, never echoed; only version/platform and exception
class/message records are emitted. No Settings, database or media is loaded.
"""

import argparse
from importlib.metadata import version
import json
import os
import platform
import subprocess
import sys


def exception_chain(error):
    chain, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append({'exception_class': type(error).__name__, 'exception_message': str(error)})
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return chain


def initialize(landmarker):
    # Match the production extractor's VIDEO/CPU defaults exactly, using only
    # reviewed packaged files. Direct APIs preserve the original native error.
    import mediapipe as mp
    from ugsl_ai_coach.assets import model_paths
    hand, pose = model_paths()
    vision = mp.tasks.vision
    if landmarker == 'HandLandmarker':
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(hand)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2)
        instance = vision.HandLandmarker.create_from_options(options)
    else:
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(pose)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1, output_segmentation_masks=False)
        instance = vision.PoseLandmarker.create_from_options(options)
    instance.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Real packaged Hand/Pose initialization; no infrastructure configuration')
    parser.add_argument('--child', choices=('HandLandmarker', 'PoseLandmarker'), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        try:
            initialize(args.child)
            chain = []
        except Exception as error:
            chain = exception_chain(error)
        print(json.dumps(chain))
        return int(bool(chain))

    print('Python:', platform.python_version())
    print('Platform:', sys.platform, platform.machine())
    print('MediaPipe:', version('mediapipe'))
    # Exclude DB/storage/service credentials, .env loading, GPU/display settings
    # and learner data. These checks deliberately validate headless CPU operation.
    env = {key: os.environ[key] for key in ('PATH', 'SYSTEMROOT', 'WINDIR', 'PYTHONPATH') if key in os.environ}
    failed = False
    for name in ('HandLandmarker', 'PoseLandmarker'):
        print('Initializing:', name, flush=True)
        try:
            result = subprocess.run([sys.executable, '-m', __package__ + '.mediapipe_check', '--child', name],
                                    env=env, capture_output=True, text=True, timeout=60)
            # Only the child's final JSON record is used; discard native output.
            chain = json.loads(result.stdout.strip().splitlines()[-1])
            failed |= result.returncode != 0 or bool(chain)
        except Exception:
            chain = [{'exception_class': 'DiagnosticProcessError',
                      'exception_message': 'Landmarker diagnostic did not return an exception record'}]
            failed = True
        for index, record in enumerate(chain):
            # Defensive redaction: never expose inherited environment values even
            # if an unexpected native error happens to include one.
            message = record['exception_message']
            for value in sorted(set(os.environ.values()), key=len, reverse=True):
                if len(value) >= 4:
                    message = message.replace(value, '[redacted]')
            print('Exception' if index == 0 else 'Chained exception', record['exception_class'], json.dumps(message))
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
