# UgSL AI Practice Coach

The Uganda Sign Language (UgSL) AI Practice Coach will eventually compare learner practice performances with expert-validated references and provide evidence-based feedback.

## Current status and scope

Milestone 1 provides a standalone Python service foundation: FastAPI application factory, versioned process health endpoint, typed environment settings, standard-library JSON application logging, and automated tests. **The current service does not establish UgSL linguistic correctness.** No storage, authentication, or frontend is implemented.

Milestone 2 adds Structured Findings Contract v1: validated Pydantic domain models and a generated JSON Schema endpoint. It defines what future analysis components may report; it produces no analysis results or feedback.

Milestone 3 adds an internal video-to-movement-observations pipeline using OpenCV and MediaPipe Tasks. It observes physical coordinates and detection coverage only. No new HTTP route, upload, webcam, or `/analyze` behavior is added.

Milestone 4 adds internal reference comparison of normalized wrist movement paths. It emits measured geometry, evidence sufficiency, and provisional MOVEMENT findings using the unchanged M2 contract. **Geometric similarity is not UgSL linguistic correctness.** No public comparison endpoint is implemented.

Milestone 5 adds internal evidence-grounded Behavior-Aware Interface (BAI) coaching with an offline deterministic provider. It explains M2 findings using validated, accessible text and annotation metadata. No public coaching endpoint, live LLM, or TTS is implemented; M6 integration has not started.

## Architecture principles

The model is a component of the AI Coach, not the AI Coach itself. Future objective CV and comparison stages will produce structured findings; a later LLM layer may explain those findings and must not independently judge sign correctness from raw video. Insufficient confidence must lead to abstention. Previous learner attempts will eventually be immutable historical records.

Requests currently flow through the FastAPI app in `src/ugsl_ai_coach/main.py` to health and contract routers under `/api/v1`. Settings are loaded when the application is created and attached to that app. Logging is configured at startup, with startup/shutdown messages and DEBUG health messages. Application logs are JSON; Uvicorn retains its own server/access logging. `core/` contains configuration and logging, `api/routes/` contains HTTP routes, `domain/analysis.py` defines the authoritative M2 contract, `cv/` provides internal M3 extraction, `comparison/` implements internal M4 reference comparison, and `coaching/` implements internal M5 explanations. Integration adapters remain future work.

## Structured Findings Contract v1

Structured Findings are the evidence-first boundary from future CV/comparison components to future coaching explanations. Identifiers trace an attempt (`ATT-` plus at least six digits), analysis (`AN-` plus at least six digits), and finding (`F-` plus at least three digits). `model_version` is trimmed and must be non-empty.

| Analysis status | Meaning and validation |
| --- | --- |
| `COMPLETED` | Sufficient evidence; score and confidence required; zero or more findings |
| `UNANALYZABLE` | Insufficient evidence, **not poor learner performance**; score absent/null, confidence required, only insufficient-evidence findings or none |
| `FAILED` | Technical/processing failure; score absent/null, confidence optional, findings empty |

Skills are exactly `HANDSHAPE`, `ORIENTATION`, `LOCATION`, `MOVEMENT`, `TIMING`, `SEQUENCE`, `MOVEMENT_RANGE`, and `BODY_POSITION`. Directional observations use `MOVEMENT` (for example expected `UPWARD`, observed `OUTWARD`); `DIRECTION` is invalid.

Finding statuses are `STRONG`, `ACCEPTABLE`, `NEEDS_IMPROVEMENT`, `WARNING`, and `INSUFFICIENT_EVIDENCE`. All except `INSUFFICIENT_EVIDENCE` require evidence with readable expected/observed strings and a finite numeric deviation. Deviation has no universal range or interpretation in M2. Severity is an explicit integer from 0 (informational) through 3 (major), never inferred from status. Body regions are `LEFT_HAND`, `RIGHT_HAND`, `BOTH_HANDS`, `LEFT_ARM`, `RIGHT_ARM`, `HEAD`, and `UPPER_BODY`. Timestamps are non-negative integers with end >= start.

Scores and confidence are finite values in [0, 1]; percentages such as 91 are rejected. A score represents internal similarity, not automatically a learner grade; M4's engineering transformation is explicitly not expert/linguistically calibrated. NaN, infinities, unexpected fields, fractional timestamps/severity, and numeric strings are rejected. Missing or null evidence is permitted for insufficient-evidence findings without fabricated observations.

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

Future work requires expert UgSL data and calibration, deliberate coverage of unsupported linguistic dimensions, and separately designed persistence/backend/frontend integration. M6 has not been started.

## M4 reference comparison

**PROVISIONAL ENGINEERING THRESHOLDS — NOT YET LINGUISTICALLY VALIDATED.** M4 v1 covers `MOVEMENT` only. It does not validate handshape, orientation, location, timing, sequence, movement range, body position, semantics, grammar, or facial/non-manual features. A geometrically different path is not necessarily invalid UgSL. Future expert calibration is required before interpreting geometric bands linguistically.

`compare_movement(reference, learner, request, policy)` accepts two existing M3 `ExtractionResult` objects and returns a frozen `ComparisonOutcome`. It never opens videos or calls MediaPipe. The caller supplies an expert-validated reference through upstream context; M4 checks technical usability, not expertise. The request contains analysis/attempt/finding IDs, a reference ID, and explicit hand correspondence. No additional dependency is used.

### Selection and correspondence

`HandCorrespondence` requires a reference model-reported label, learner model-reported label, caller-established anatomical body region, and non-empty establishment basis. Absent correspondence causes abstention. This attestation does not independently verify identity: upstream metadata/calibration must establish it. M4 never selects the cheapest hand pairing, swaps labels, mirrors coordinates, or guesses identity from image position. Explicitly different reference/learner labels are allowed only when the caller supplies their correspondence.

Use normalized hand landmark **0 (wrist)** x/y only. No raw z, hand-landmark averaging, trajectory recentering, or new normalization is applied. Every selected trajectory retains sampled timestamps/frame indices and availability states. Missing hands, duplicate matching labels, missing/low label confidence, unavailable normalization, or missing normalized wrists are gaps. Inconsistent raw/normalized detection indexing is invalid caller input. No zeros, coordinate carry-forward, or interpolation fill gaps.

### Eligibility and confidence

Count actual usable wrist observations and their fraction of all sampled frames separately for reference and learner. Both must meet the configured count/coverage thresholds before DTW. Coverage is frame-based, not duration-weighted. Confidence is the auditable evidence-sufficiency surrogate:

```text
min(reference_coverage, learner_coverage,
    min(1, reference_usable_count / minimum_usable_observations),
    min(1, learner_usable_count / minimum_usable_observations))
```

It contains no coordinates, distance, similarity, or status-band values. Thus fully observed different paths can have low similarity and confidence 1. Confidence is not a calibrated probability of linguistic correctness. No established correspondence yields confidence 0. Missing observations affect confidence; maximum gaps and observed duration remain descriptive metrics, not timing judgments. DTW compares actual observed sequences across gaps, without claiming to reconstruct unobserved motion.

### Alignment, metrics, and similarity

Exact global DTW aligns the available normalized x/y observations using Euclidean distance in shoulder-width units. It minimizes cumulative local cost, permitting diagonal, reference-advance, and learner-advance steps. Ties prefer those steps in that order. Cost rows use O(learner count) memory; a bounded one-byte-per-cell predecessor grid reconstructs the complete path. Time is O(reference count × learner count). Empty, missing, unordered, or non-finite input is rejected.

Returned metrics include reference/learner sampled and usable counts, usable coverage, first-to-last usable duration, largest usable timestamp gap, cumulative DTW cost, alignment path length, cumulative cost divided by path length, engineering similarity, and evidence confidence. Mean path distance is the mean along the **minimum-cumulative-cost path**, not a separate minimum-mean optimization or a mathematical distance metric guarantee.

Engineering movement similarity is `scale / (scale + mean_path_distance)`: distance 0 gives 1; increasing distance monotonically reduces similarity. The configurable scale is the distance giving similarity 0.5. Numerically stable evaluation avoids overflow. This simple mapping is a dimensionless engineering convenience, not expert-calibrated correctness. DTW accommodates speed/length differences and does not measure linguistic timing correctness.

### Default versioned policy

Algorithm: `movement-wrist-dtw-v1`. Policy: `provisional-engineering-v1`. The policy snapshot has `linguistically_validated=False`; all configurable values and band mappings are retained in the outcome. M2 `model_version` combines algorithm/policy version identifiers. Supply a new policy version when replacing thresholds/mappings.

| Setting | Default | Meaning |
| --- | --- | --- |
| Minimum usable observations | 8 per trajectory | Eligibility; provisional, not expert calibrated |
| Reference coverage minimum | 0.8 | Eligibility; provisional, not expert calibrated |
| Learner coverage minimum | 0.8 | Eligibility; provisional, not expert calibrated |
| Handedness confidence minimum | 0.5 | Per-frame label filter; provisional, not anatomical calibration |
| Similarity distance scale | 1 shoulder width | Engineering mapping; not expert calibrated |
| Strong maximum mean distance | 0.1 shoulder widths, inclusive | Provisional `STRONG`, severity 0 |
| Acceptable maximum mean distance | 0.3 shoulder widths, inclusive | Provisional `ACCEPTABLE`, severity 0 |
| Distance above acceptable maximum | >0.3 shoulder widths | Provisional `NEEDS_IMPROVEMENT`, severity 1 |
| Maximum alignment cells | 1,000,000 | Processing resource limit, not a linguistic threshold |

All status and severity mappings live in the policy and are replaceable within M2's existing enums/ranges. Insufficient evidence always uses severity 0, without a performance band. No `WARNING` or unsupported-skill findings are manufactured. Count/coverage/hand-label thresholds and status bands are engineering defaults only; none represents expert-validated acceptable UgSL variation.

### Results, failures, and traceability

`COMPLETED` has a similarity score, separate evidence confidence, and one MOVEMENT finding whose evidence reports measured mean DTW distance, aligned-pair count, and the provisional threshold. It never invents directional claims. Finding start/end enclose actual usable learner observations (floor first timestamp, ceil last timestamp to satisfy M2 integer milliseconds).

`UNANALYZABLE` has no score and a typed reason distinguishing absent correspondence, insufficient reference evidence, insufficient learner evidence, or both. If an actual learner evidence interval exists, emit one severity-0 `INSUFFICIENT_EVIDENCE` finding with no fabricated evidence; otherwise emit no findings/timestamps.

`FAILED` represents only known processing failures: alignment resource-budget exhaustion or non-finite numerical computation. Score/confidence are null and findings empty. Invalid caller/configuration data raises typed validation/input errors; unexpected programming exceptions propagate rather than becoming learner findings. Technical reason codes remain in the internal outcome, not a new M2 error payload.

The outcome retains request/reference identifiers, policy and lightweight M3 metadata contexts, both trajectories with gaps, eligibility, metrics linked by finding ID, and the alignment's sequence/frame indices and local distances. This traces findings back to actual source observations without putting arrays in evidence strings or duplicating full ExtractionResults.

No learner/reference recordings or datasets are bundled or downloaded. Tests use typed synthetic M3 observations, including actual M3 normalization for translation/scale checks. Comparison performs no persistence, raw-array logging, network access, public API exposure, or coaching.

## M5 evidence-grounded BAI coaching

**M3 observes. M4 measures. M5 explains.** M5 may explain existing evidence; it never creates evidence. Its only analysis input is a validated M2 `StructuredAnalysisResult`. It never reads video, landmarks, trajectories, DTW internals, M4 reason codes, or arbitrary application data, and never reruns extraction/comparison or calculates a score.

```text
M3 observation -> M4 comparison -> M2 StructuredAnalysisResult
  -> ground_analysis -> CoachingContext -> CoachingProvider
  -> validate_feedback -> CoachingFeedback -> future platform integration
```

Example internal use (no route, persistence, audio, or API key needed):

```python
from ugsl_ai_coach.coaching.engine import generate_feedback

# analysis is an existing validated M2 StructuredAnalysisResult.
feedback = generate_feedback(analysis, feedback_id="FB-review-example")
```

The engine generates a UUID-based feedback ID when one is not supplied. Tests inject IDs. Grounding revalidates M2 input, rejects ambiguous duplicate finding IDs, ignores unsupported skills, and retains unchanged movement finding IDs, body regions, timestamps, severity and confidence. It derives safe explanations from source status without interpreting deviation or copying free-form evidence strings/model versions into provider input. The context contains confidence as technical evidence-sufficiency metadata, authorized points, status, one action, versioned constraints and safe wording; it excludes engineering similarity. Confidence is not a probability of correctness.

| Source state/finding | Coaching behavior |
| --- | --- |
| `COMPLETED` | Analysis finished, not automatically good performance. Empty or unsupported-only findings produce no strengths/corrections. |
| `STRONG` movement | Cautious, traceable movement strength; no claim of linguistic correctness. |
| `ACCEPTABLE` movement | Neutral observation within the provisional comparison range; no failure language or exaggerated praise. |
| `NEEDS_IMPROVEMENT` movement | Traceable movement path difference and optional reference review; no invented directional instruction or severity change. |
| `WARNING` movement | Acknowledge an upstream warning without guessing its meaning/cause or turning it into a correction. |
| `INSUFFICIENT_EVIDENCE` movement | Describe missing reliable evidence, never poor performance. |
| `UNANALYZABLE` | No performance judgment. Generic optional recording retry, with any supported insufficient-evidence interval. |
| `FAILED` | No performance judgment or technical error text. Generic optional analysis retry later. |

M2 does not carry M4's typed failure reasons. M5 therefore cannot infer hand visibility, framing, lighting, resource limits or other causes from these states. **When the system does not know, the coaching response communicates uncertainty rather than inventing an answer.** Confidence below full evidence sufficiency uses explicitly limited, tentative wording; no new score/confidence threshold is introduced. Any limited or warning/insufficient movement evidence remains acknowledged in the summary even if another observation is prioritized.

### BAI policy and contracts

**Guide the learner toward clarity, not toward obedience.** `policy.py` centralizes `bai-coaching-v1`, `movement-coaching-v1`, authorized templates, action wording and provider-neutral constraints. Learner text gives an observation, its limited meaning and one optional action. Actions start with “You can”; encouragement is process-oriented rather than unsupported praise. There are no urgency, shame, identity judgments, threats, progression locks or course-completion decisions. Real movement differences remain explicit; emotional safety does not replace truthful feedback with reassurance.

To bound cognitive load, grounding selects at most one strength, correction and observation, ordered by supplied severity descending, confidence descending and finding ID for deterministic ties. Source severity/confidence are retained, never recalculated. Lower-priority findings remain in the original analysis; feedback is a focused view rather than a replacement evidence record.

All M5 models reject extra fields and are frozen; collections are tuples. `CoachingFeedback` contains validated feedback/attempt/analysis IDs, upstream status, summary, strengths, corrections, neutral/uncertainty observations, a typed `RecommendedAction`, encouragement, optional `audio_text`, annotations and `ProviderMetadata`. `CoachingPoint` traces each strength/correction/observation to a source finding and its skill/body/interval, with message and reason. `AuthorizedFact` adds unchanged source status/severity/confidence inside the context. Actions have only advisory semantics: `REVIEW_MOVEMENT`, `RETRY_ATTEMPT`, `RETRY_CAPTURE`, `TRY_LATER`.

```text
Learner feedback -> correction/strength -> finding_id
  -> StructuredFinding -> M4 evidence -> source observations
```

`VisualAnnotation` retains source finding ID, MOVEMENT skill, body region, exact interval and an already-accessible point label. No video rendering or invented timestamps occurs. Annotations are metadata for a future frontend; they add no facts absent from text. `accessible_text` provides the complete reading order: summary, points and reasons, next action, encouragement. Audio is absent by default. The deterministic provider can optionally supply `audio_text`, which must equal that complete text exactly. No TTS or audio files exist; nothing requires audio.

### Providers and fail-closed validation

`CoachingProvider` is a vendor-neutral protocol taking only `CoachingContext` and returning `CoachingFeedback`. `DeterministicCoachingProvider` assembles the authorized facts offline. Provider metadata records provider type, policy version and template version outside learner text. No dependencies, credentials, network, LLM SDK or vendor integration are added.

V1 deliberately uses a **closed authorized vocabulary**. Final validation revalidates the typed result and requires exact agreement with the trusted context for all identifiers, status, source points, categories, skill/body/timestamps, learner wording, action and encouragement. It also checks exact annotations, provider metadata and optional audio equivalence. Providers cannot omit the prioritized difference, add praise, alter severity through wording, expand/shrink an interval, move a strength into corrections, or hide a claim in summary/action/audio. Violations raise `GroundingViolation` (or schema validation errors); no feedback is returned and no hallucinated claim is silently repaired. Unexpected provider exceptions propagate to the internal caller, not to learner-facing text. There is no automatic fallback masking a provider failure.

A future LLM provider must obey these same constraints. Arbitrary paraphrases are intentionally rejected in v1: unrestricted language would require separately designed semantic validation and a versioned policy change, not merely schema compliance. The central constraints require evidence-only explanations, uncertainty, unchanged source attribution/severity/times, agency and respectful language. M5 never claims handshape, orientation, location, timing, sequence, movement range, body position, direction, grammar, meaning, facial/non-manual correctness or complete UgSL correctness. Engineering similarity is never presented as a grade, percentage, mastery, pass threshold or linguistic accuracy.

Coaching tests use only typed M2 fixtures, including malicious provider output, incomplete evidence, every state, unsupported skills, source/annotation tampering, closed-language BAI rules and audio equivalence. A fresh-process test blocks network, CV/comparison, MediaPipe, external AI and TTS imports while exercising coaching. Existing M1–M4 tests remain unchanged. M5 retains data in memory only and adds no logging of learner evidence or persistence.
