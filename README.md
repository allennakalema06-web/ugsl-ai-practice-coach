# UgSL AI Practice Coach

The Uganda Sign Language (UgSL) AI Practice Coach will eventually compare learner practice performances with expert-validated references and provide evidence-based feedback.

## Current status and scope

Milestone 1 provides a standalone Python service foundation: FastAPI application factory, versioned process health endpoint, typed environment settings, standard-library JSON application logging, and automated tests. **The current service does not evaluate UgSL sign correctness.** No comparison, scoring, coaching, storage, authentication, or frontend is implemented.

Milestone 2 adds Structured Findings Contract v1: validated Pydantic domain models and a generated JSON Schema endpoint. It defines what future analysis components may report; it produces no analysis results or feedback.

Milestone 3 adds an internal video-to-movement-observations pipeline using OpenCV and MediaPipe Tasks. It observes physical coordinates and detection coverage only. No new HTTP route, upload, webcam, or `/analyze` behavior is added.

## Architecture principles

The model is a component of the AI Coach, not the AI Coach itself. Future objective CV and comparison stages will produce structured findings; a later LLM layer may explain those findings and must not independently judge sign correctness from raw video. Insufficient confidence must lead to abstention. Previous learner attempts will eventually be immutable historical records.

Requests currently flow through the FastAPI app in `src/ugsl_ai_coach/main.py` to health and contract routers under `/api/v1`. Settings are loaded when the application is created and attached to that app. Logging is configured at startup, with startup/shutdown messages and DEBUG health messages. Application logs are JSON; Uvicorn retains its own server/access logging. `core/` contains configuration and logging, `api/routes/` contains HTTP routes, `domain/analysis.py` defines the authoritative M2 contract, and `cv/` provides internal M3 extraction. Future comparison, coaching, and adapter modules will be added when needed.

## Structured Findings Contract v1

Structured Findings are the evidence-first boundary from future CV/comparison components to future coaching explanations. Identifiers trace an attempt (`ATT-` plus at least six digits), analysis (`AN-` plus at least six digits), and finding (`F-` plus at least three digits). `model_version` is trimmed and must be non-empty.

| Analysis status | Meaning and validation |
| --- | --- |
| `COMPLETED` | Sufficient evidence; score and confidence required; zero or more findings |
| `UNANALYZABLE` | Insufficient evidence, **not poor learner performance**; score absent/null, confidence required, only insufficient-evidence findings or none |
| `FAILED` | Technical/processing failure; score absent/null, confidence optional, findings empty |

Skills are exactly `HANDSHAPE`, `ORIENTATION`, `LOCATION`, `MOVEMENT`, `TIMING`, `SEQUENCE`, `MOVEMENT_RANGE`, and `BODY_POSITION`. Directional observations use `MOVEMENT` (for example expected `UPWARD`, observed `OUTWARD`); `DIRECTION` is invalid.

Finding statuses are `STRONG`, `ACCEPTABLE`, `NEEDS_IMPROVEMENT`, `WARNING`, and `INSUFFICIENT_EVIDENCE`. All except `INSUFFICIENT_EVIDENCE` require evidence with readable expected/observed strings and a finite numeric deviation. Deviation has no universal range or interpretation in M2. Severity is an explicit integer from 0 (informational) through 3 (major), never inferred from status. Body regions are `LEFT_HAND`, `RIGHT_HAND`, `BOTH_HANDS`, `LEFT_ARM`, `RIGHT_ARM`, `HEAD`, and `UPPER_BODY`. Timestamps are non-negative integers with end >= start.

Scores and confidence are finite values in [0, 1]; percentages such as 91 are rejected. The score is an internal calibrated similarity measure, not automatically a learner grade. NaN, infinities, unexpected fields, fractional timestamps/severity, and numeric strings are rejected. Missing or null evidence is permitted for insufficient-evidence findings without fabricated observations.

Contract models are frozen, including nested evidence/findings; findings use an immutable tuple in Python and a JSON array on the wire. This protects validated objects from ordinary accidental mutation, **not database-level historical immutability**. No persistence exists. Construct contracts through normal Pydantic validation; unchecked construction/copy APIs are not a validation boundary.

`GET http://127.0.0.1:8000/api/v1/contracts/analysis` returns JSON Schema generated directly from `StructuredAnalysisResult`. Field descriptions document conditional requirements; cross-field validators remain authoritative because generated JSON Schema does not encode those validators as conditional schema rules. There is no `/analyze` endpoint.

## Local setup (Windows PowerShell)

Prerequisites: Python 3.13 and its Windows `py` launcher. From the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Activation is optional: use `.\.venv\Scripts\python.exe` in place of `python` if PowerShell policy prevents activation. Runtime-only installation uses `python -m pip install -e .`.

Optionally copy the example settings:

```powershell
Copy-Item .env.example .env
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `UGSL_SERVICE_NAME` | `ugsl-ai-practice-coach` | Service identifier returned by health |
| `UGSL_ENVIRONMENT` | `development` | Deployment environment label |
| `UGSL_LOG_LEVEL` | `INFO` | Application logging: DEBUG, INFO, WARNING, ERROR, CRITICAL |

Environment variables override `.env` values. `.env` is optional and read relative to the working directory. No secrets or provider credentials are required.

## Run the API

From the repository root (activation not required):

```powershell
.\.venv\Scripts\python.exe -m uvicorn ugsl_ai_coach.main:app --host 127.0.0.1 --port 8000
```

Add `--reload` for local development if desired. Stop with Ctrl+C.

`GET http://127.0.0.1:8000/api/v1/health` returns HTTP 200 and, with default settings:

```json
{"status": "ok", "service": "ugsl-ai-practice-coach"}
```

This checks only that the service responds; it calls no external systems. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Tests use an in-process HTTP client and controlled settings. They need no external network, API keys, webcam, or database. CV unit tests generate tiny synthetic MJPEG videos in pytest's temporary directory and inject landmark observations. The optional MediaPipe integration smoke uses blank frames and skips if local model assets are absent. Run it explicitly with `python -m pytest -m integration`; unit/API tests alone use `python -m pytest -m "not integration"`. Development dependencies remain pytest and HTTPX. No separate lint/format tool is introduced.

## M3 extraction pipeline

`inspect_video` validates a local regular file and scans its decoded frames without retaining pixels. `sample_frames` selects real frames by timestamp. `MediaPipeExtractor` runs separate Hand Landmarker and Pose Landmarker Tasks in VIDEO mode on CPU. `normalize` transforms planar coordinates. `extract_video` returns one frozen `ExtractionResult` with video/sampling metadata, normalization configuration, ordered frame observations, raw/normalized landmarks, gaps, and descriptive coverage counts. Frame index plus landmark index and reported hand label identify observations over time; the hand list retains detection order, not persistent tracking IDs. M4 must resolve identity ambiguity before comparison.

M3 never constructs M2 Structured Findings, severity, similarity, direction classifications, or learner feedback. M4 will compare the underlying time series with expert references and determine whether evidence supports any findings.

### Dependencies and model assets

The tested Windows x64 environment uses Python 3.13.15, `mediapipe==1.0.1`, `opencv-contrib-python==5.0.0.93`, and `numpy==2.5.3`. OpenCV contrib supplies `cv2` and satisfies MediaPipe's dependency; do not also install `opencv-python` or another OpenCV distribution into the same environment. NumPy holds decoded image arrays and transfers them to MediaPipe. Existing API dependencies are unchanged. MediaPipe brings transitive dependencies including matplotlib and sounddevice; this service uses neither plotting nor audio capture.

Tasks requires local model bundles. Provision them explicitly once from Google's versioned model URLs; neither runtime code nor tests downloads anything:

```powershell
New-Item -ItemType Directory -Path .models -Force | Out-Null
Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' -OutFile '.models/hand_landmarker.task'
Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task' -OutFile '.models/pose_landmarker_lite.task'
Get-FileHash .models/*.task -Algorithm SHA256
```

SHA256 of the tested version-1 assets:

- `hand_landmarker.task`: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`
- `pose_landmarker_lite.task`: `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`

Assets are ignored and not bundled. Custom asset paths can be supplied to the extractor; model upgrades require deliberate verification. See Google's [Hand Tasks guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python) and [Pose Tasks guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python).

Example internal use (provide your own local video; no automatic persistence):

```python
from ugsl_ai_coach.cv.landmarks import MediaPipeExtractor
from ugsl_ai_coach.cv.models import SamplingConfig
from ugsl_ai_coach.cv.pipeline import extract_video

with MediaPipeExtractor('.models/hand_landmarker.task', '.models/pose_landmarker_lite.task') as extractor:
    result = extract_video('local-data/practice.mp4', extractor, SamplingConfig(target_fps=60))
```

### Video and sampling assumptions

Supported files must decode through the installed OpenCV backend into fixed-size, upright, 8-bit BGR frames. Codec availability is backend-dependent. The pipeline applies no mirroring, orientation correction, or resizing. Use a single signer; multi-person identity association is outside M3. Technical safety bounds are positive dimensions up to 32768 per axis and finite FPS in (0, 1000]; these are not educational recording recommendations.

Videos with no usable frames, invalid metadata, non-monotonic/invalid timestamps, or premature decode termination relative to reported frame count raise `VideoError`. Files are inspected and then decoded sequentially a second time; changes between passes are rejected when dimensions, counts, or timestamp ordering become inconsistent. Do not modify the input during extraction.

Sampling defaults to a conservative 60 Hz target. Select the first available frame at or after each target time, skipping missed target slots without repeating frames. Lower-FPS sources retain their actual frames; no interpolation or synthetic frames. Decoder timestamps are rebased to the first frame and retained as floating-point milliseconds, including irregular intervals. If the backend reports only zero timestamps, use an explicitly labeled `fps_estimate` based on frame index and FPS; that fallback assumes constant frame rate and cannot recover variable-rate timing. Duration is estimated through the last frame plus one nominal frame interval. MediaPipe receives rounded integer milliseconds and rejects collisions rather than fabricating time.

### Coordinates, handedness, and gaps

Raw x/y are MediaPipe image-relative coordinates (x scaled by width, y by height), with origin at image top-left and y increasing downward. All 21 landmarks of each detected hand are retained. Pose retention is deliberately limited to anatomical shoulder, elbow, and wrist indices 11–16; no face mesh or facial-expression analysis is used.

Raw z is preserved: hand depth is relative to its wrist; pose depth is relative to the hip midpoint. These model-relative depth estimates are not calibrated metric depth and do not share a common origin. Shoulder normalization therefore emits x/y only; raw z remains available separately. Available pose visibility/presence and hand handedness confidence are retained. Handedness confidence is a label probability, not a per-landmark detection score. Unavailable quality values remain null.

Frames remain unmirrored. `reported_handedness` preserves the Tasks classifier's `Left`/`Right` output through one centralized mapping, not image-left/image-right. The output convention explicitly marks these as **model-reported labels**, not independently verified anatomical identities. No inversion is imported from the legacy mirrored-selfie Solutions API. Duplicate labels retain both detections; classification ambiguity is not silently resolved. Anatomical mapping/calibration and persistent identity require deliberate verification before M4 uses them.

No detection is an empty hand/pose tuple; a missing shoulder is an absent index. Missing data is never zero-filled, carried forward, or interpolated. Ordered frame observations form trajectories directly without a duplicate trajectory matrix: M4 can look up an available landmark at each timestamp and see every gap.

### Normalization and coverage

Convert raw x/y into common image-width units as `(x, y * height / width)` before calculating shoulder distances. Origin is the midpoint of shoulder indices 11 and 12; scale is their Euclidean distance. Each point becomes `(point - origin) / scale`. This accounts for frame aspect ratio and reduces translation/scale sensitivity without altering raw coordinates or clamping transformed values.

Missing shoulders, supplied anchor visibility/presence below the configurable default 0.5, or shoulder width <= the configurable numerical tolerance 1e-6 yield explicit per-frame unavailability states with no transformed points. Non-finite coordinates are rejected at model validation. Failed normalization does not discard raw observations or invalidate the entire video. This is geometric preprocessing, not a judgment of learner skill.

Coverage reports sampled frames, frames with pose, frames with each model-reported hand label, and frames normalized. A frame with two identically labeled hands counts once for that label. Counts describe detection availability, never performance quality.

### Privacy

The pipeline reads local files and keeps observations in memory; it does not save videos, extracted frames, or landmark arrays, and does not log their contents. Keep sensitive inputs under ignored `local-data/`; `.models/`, extracted-frame directories, and common video extensions are ignored. Do not commit learner recordings or derived biometric data. Results contain no source-file path.

MediaPipe processes input on device. Its [upstream privacy notice](https://pypi.org/project/mediapipe/) states that Tasks may send performance/utilization metrics; local input processing must not be conflated with a guarantee of no library telemetry. The initialization smoke succeeds with network access restricted after dependencies/assets are installed.

## Future milestones

M4 will address expert-reference comparison of M3 trajectories, calibration, confidence/abstention, and evidence-grounded Structured Findings. Later work may add coaching explanations and persistence/backend/frontend integration. No comparison or learner correctness evaluation exists in M3.
