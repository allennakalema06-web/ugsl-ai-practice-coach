# UgSL AI Practice Coach

The Uganda Sign Language (UgSL) AI Practice Coach will eventually compare learner practice performances with expert-validated references and provide evidence-based feedback.

## Current status and scope

Milestone 1 provides a standalone Python service foundation: FastAPI application factory, versioned process health endpoint, typed environment settings, standard-library JSON application logging, and automated tests. **The current service does not establish UgSL linguistic correctness.** No storage, authentication, or frontend is implemented.

Milestone 2 adds Structured Findings Contract v1: validated Pydantic domain models and a generated JSON Schema endpoint. It defines what future analysis components may report; it produces no analysis results or feedback.

Milestone 3 adds an internal video-to-movement-observations pipeline using OpenCV and MediaPipe Tasks. It observes physical coordinates and detection coverage only. No new HTTP route, upload, webcam, or `/analyze` behavior is added.

Milestone 4 adds internal reference comparison of normalized wrist movement paths. It emits measured geometry, evidence sufficiency, and provisional MOVEMENT findings using the unchanged M2 contract. **Geometric similarity is not UgSL linguistic correctness.** No public comparison endpoint is implemented.

Milestone 5 adds internal evidence-grounded Behavior-Aware Interface (BAI) coaching with an offline deterministic provider. It explains M2 findings using validated, accessible text and annotation metadata. No public coaching endpoint, live LLM, or TTS is implemented.

Milestone 6A adds a transport-independent backend integration contract, immutable job snapshots, lifecycle coordination and technology-neutral ports. Milestone 6B adds atomic job/work acceptance, lease-based recovery and atomic analysis/work completion semantics. Concrete infrastructure and M6C+ remain future work.

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

M6F packages these exact version-1 assets under `ugsl_ai_coach.assets`, with checksums, provenance and Apache 2.0 license notices. The development `.models/` directory remains ignored. Production deployment commands default to the packaged assets; explicit custom paths remain supported and require matching reference-profile model hashes. Model upgrades require deliberate verification. See Google's [Hand Tasks guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python) and [Pose Tasks guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python).

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

Future work requires expert UgSL data and calibration, deliberate coverage of unsupported linguistic dimensions, and separately designed persistence/backend/frontend integration. M6A/M6B define integration and durability contracts; M6C+ has not been started.

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

## M6A Integration Contract

These are **M6 integration engineering decisions**, not claims about an earlier external API specification. No core product/API source documents are bundled in this repository; this section records the implemented boundary.

`integration/` provides frozen, extra-forbidding `AnalysisSubmission`, `AnalysisJob`, `JobAcceptance` and separate `CoachingRecord` models. A submission contains the existing M2 `attempt_id` plus strict, trimmed, non-empty opaque `learner_video_ref`, `reference_profile_ref` and `idempotency_key` strings. References are not interpreted as paths, URLs or storage-provider keys. Jobs retain these values and an M2-format `analysis_id`; terminal results reuse `StructuredAnalysisResult` directly. `CoachingRecord` contains an existing typed M5 `CoachingFeedback` with matching analysis/attempt identifiers.

```text
UgSL backend -> AnalysisIntegrationService.submit
  -> atomic AnalysisJobRepository.accept -> AnalysisDispatcher
future worker -> mark_processing -> complete(existing M2 result) -> immutable analysis evidence
later M5 output -> CoachingIntegrationService.record_feedback -> separate append-only coaching record
```

`SUBMITTED` and `PROCESSING` are analysis integration states only. M2 terminal statuses are unchanged. The only transitions are `SUBMITTED -> PROCESSING -> COMPLETED | UNANALYZABLE | FAILED`. Non-terminal jobs contain no analysis result; coaching is never a job field. Terminal jobs require a matching M2 result; state, analysis ID and attempt ID must agree. `COMPLETED` means the analysis stage completed successfully, not that downstream coaching or presentation has finished. `UNANALYZABLE` and `FAILED` also remain analysis outcomes. The service validates supplied results; it does not execute M3/M4 or generate coaching.

Terminal analysis snapshots cannot transition again, including same-state updates, result replacement or retries. Future retries create new jobs rather than rewrite earlier evidence. Frozen Python objects protect ordinary in-memory mutation; future adapters must enforce durable historical immutability. Analysis ID allocation is injected and preserves `AN-[0-9]{6,}`; no production ID algorithm is selected.

Coaching can be persisted later without changing the terminal job or its StructuredAnalysisResult. `CoachingIntegrationService` reads existing terminal analysis evidence, verifies all analysis/attempt identifiers, and applies the unchanged M5 grounding/accessibility validator, including source findings, intervals, body regions, authorized claims and audio equivalence. It then appends a separate `CoachingRecord`; no provider is invoked and no evidence or lifecycle state is changed. Missing/non-terminal analysis rejects coaching persistence. No coaching lifecycle state is added.

`CoachingRecordRepository` is a separate technology-neutral append-only port: one logical coaching artifact per analysis in M6A. Exactly identical writes, including feedback ID, metadata and optional audio representation, return the existing record without duplication. Conflicting output raises `FeedbackConflict` and never overwrites history. Future adapters must enforce atomicity for racing writes; test fakes are sequential and add no database locking. Versioned alternative coaching artifacts would require a later explicit contract decision.

Idempotency compares the canonical trimmed submission. Same key plus same payload returns the existing job in its current state without duplicate work or redispatch. Same key plus changed attempt/video/profile raises `IdempotencyConflict`. `AnalysisJobRepository.accept` must atomically enforce this rule and analysis-ID uniqueness, including racing submissions. `compare_and_set` must atomically validate transitions and match the full expected snapshot, rejecting stale updates. Test-only deterministic fakes illustrate these obligations; no production repository is included.

`AnalysisJobRepository`, `CoachingRecordRepository`, `AnalysisDispatcher` and `AnalysisIdFactory` are technology-neutral protocols. Database, queue, object storage, deployment and service-authentication providers remain undecided. Backend authentication/authorization remains authoritative; possessing an ID grants no access. Key namespace/tenant scoping belongs to later authorized adapter wiring.

Persistence and dispatch are separate ports: M6A does **not** guarantee durable handoff or exactly-once worker execution. `SUBMITTED` does not guarantee successful dispatch. Dispatch errors propagate and may leave an accepted `SUBMITTED` job; repeating its submission returns that job rather than automatically dispatching again. Later M6 infrastructure wiring must resolve durable handoff/recovery and the acceptance/dispatch failure window before production use. No automatic retries or recovery infrastructure are implemented.

States describe the system, not learner quality: processing provides no performance conclusion, `UNANALYZABLE` means insufficient evidence, and `FAILED` means the analysis system could not complete. M6A adds no learner-facing copy, coercion, grades or progress percentages. M5 remains responsible for complete text/visual feedback and equivalent optional audio.

M6A adds no mounted HTTP endpoint, media I/O/persistence/logging, raw frames/landmarks/DTW arrays, credentials, production infrastructure, LLM, TTS or frontend behavior. Tests use synthetic references and injected adapters; M1–M5 remain unchanged.

### M6B durable work handoff and persistence semantics

These M6 engineering decisions extend the persisted-evidence pipeline without selecting a database, queue or deployment provider. `integration/handoff/` is the M6B entry point. The M6A `AnalysisIntegrationService` remains a compatibility coordinator for its earlier contract; its separate accept/dispatch path does **not** provide M6B guarantees and must not be used for durable M6B submission. Neither service or test adapter is wired into application startup.

`AnalysisHandoffService.submit` makes one `AnalysisPersistence.accept` call committing an `AnalysisJob(SUBMITTED)` and `AnalysisWorkItem(PENDING)` together. No two independent application writes and no immediate dispatch call are required. A successfully accepted new submission must have its recoverable work record; job-only acceptance is not part of this port. Production adapters must make the commit durable across process restart. Same idempotency key plus canonical payload returns the existing job/work pair at any stage; conflicting payload or colliding analysis IDs are rejected without rewriting history.

```text
backend submission -> atomic job + pending-work commit
  -> future worker claim_next -> begin -> PROCESSING
  -> existing M2 result -> atomic terminal-job + completed-work commit
  -> later, separate immutable CoachingRecord
```

Work delivery state is separate from analysis state: `PENDING -> CLAIMED -> COMPLETED`. An expired `CLAIMED` item is directly reclaimed into a new `CLAIMED` generation, without changing analysis state or IDs. Work has only analysis/attempt IDs, opaque video/reference values, state, `delivery_count`, `claimed_at_ms` and `lease_expires_at_ms`. It contains no M2 result, score, coaching, raw media or learner PII. Pair validation requires matching identities/references and terminal analysis if and only if work is completed. Invalid/missing paired repository state is rejected, not repaired.

All time values are explicit strict non-negative UTC epoch milliseconds; lease durations are strict positive integers. Tests inject values without sleeping. Future adapters must provide trusted authoritative time shared by workers, not accept untrusted client clocks. A lease is active during `claimed_at_ms <= now_ms < lease_expires_at_ms`; at expiry it can be recovered. Claiming must be atomic so only one active delivery exists. Every claim/reclaim increments `delivery_count`, which also fences stale workers: begin and finish must match the complete current claim and an unexpired lease inside the persistence transaction. This count measures infrastructure deliveries, **not learner retries, practice attempts or learner failures**.

Beginning a claimed job changes `SUBMITTED -> PROCESSING`; reclaim of already-processing work resumes the same job without rewriting its lifecycle. Expiry/recovery never returns the job to SUBMITTED or creates a terminal outcome. Worker crashes and optional wakeup/dispatch failures retain recoverable work and must not synthesize M2 `FAILED`. Analysis `FAILED` is a valid terminal result only when supplied by the analysis pipeline, and closes work just like COMPLETED or UNANALYZABLE.

Finalization atomically preserves the supplied matching M2 result and marks work COMPLETED. Lease fields are cleared while delivery count remains. Completed work is never reopened; repeated or stale completion attempts are rejected. If acknowledgement is lost after commit, callers can inspect the terminal job/work pair rather than rewrite evidence. Execution may repeat after lease expiry; this is recoverable delivery, not a guarantee of exactly-once external execution. Lease renewal, scheduling fairness, retry limits and operational recovery policy are deliberately not implemented. Long-running work whose lease expires must be reclaimed before finalization; adapters must prevent clock regressions from granting invalid ownership.

`AnalysisPersistence` exposes only checked reads, atomic accept, claim, begin and finish. Its `AnalysisJobReader` view lets the unchanged coaching service validate later feedback against terminal evidence. Coaching remains separately append-only; no job/work mutation or coaching generation is introduced. The fake persistence adapter lives under `tests/integration_handoff/`, is sequential/non-production and **not durable**. JSON snapshots simulate committed-state restart solely to test the contract, including crashes before worker delivery, lease expiry and acknowledgement loss. It is never installed or registered at startup.

The pending-work representation closes M6A's logical acceptance/dispatch gap; actual durable storage and production recovery require a later persistence adapter satisfying these atomic obligations. No concrete DB, queue, object storage, auth mechanism, network worker, media I/O, HTTP endpoint, LLM, TTS or dependency is added. The full M1–M6A suite remains unchanged and must continue to pass.

### M6C production persistence and worker runtime

**PostgreSQL is a new M6C engineering decision**, not a database selection attributed to the original product documents. One PostgreSQL system now handles durable analysis jobs, work handoff, idempotency, leases, terminal evidence and separate coaching records. This avoids a separate broker for the current workload. Psycopg 3 (`psycopg[binary]>=3.2,<4.0`) executes explicit parameterized SQL; no ORM, pool or broker is introduced. Use a currently supported PostgreSQL release (14 or newer); no experimental features are required. This adapter implements the existing M6A/M6B contracts without changing M1–M6B models or policies.

`infrastructure/postgres/` contains the connection factory, checked serialization, explicit migrations, `PostgresAnalysisPersistence`, `PostgresAnalysisIdFactory` and `PostgresCoachingRepository`. Jobs and work retain searchable IDs, references, states and operational timestamps as columns. M2 results and M5 feedback are separate validated JSONB artifacts. IDs use the existing `AN-[0-9]{6,}` format; the production factory renders a random UUID as decimal digits. Database constraints enforce unique identities/keys, matching references, valid states and delivery metadata. Deferred constraint triggers forbid job-only acceptance or half-terminal commits. History triggers prohibit reopening terminal jobs/completed work, deleting history or replacing coaching. Invalid stored JSON, missing pairs and inconsistent identifiers raise `CorruptPersistence`; reads never repair evidence.

Acceptance inserts SUBMITTED job and PENDING work in one transaction. A unique idempotency key serializes racing accepts: an identical canonical submission returns its existing pair, while changed payload raises `IdempotencyConflict`. Claims lock eligible PENDING or expired CLAIMED rows with `FOR UPDATE SKIP LOCKED` and increment the delivery generation atomically. Begin and finish lock work and require the exact current, unexpired claim. Begin moves SUBMITTED to PROCESSING or resumes existing PROCESSING. Finish atomically commits the supplied matching M2 terminal result and COMPLETED work, clearing lease fields. Narrow state/generation/expiry predicates and database triggers protect immutable evidence. Coaching is appended independently, with full unchanged M5 grounding/accessibility validation; identical writes deduplicate and conflicting writes raise `FeedbackConflict`.

**Database time is authoritative for production leases.** The adapter obtains `clock_timestamp()` after acquiring locks and stores exact millisecond TIMESTAMPTZ values, exposing existing M6B epoch-millisecond fields. Optional `now_ms` arguments retain the M6B port signature but are schema-validated hints and never grant ownership. `AnalysisHandoffService.submit` can use this adapter; its deterministic clock-specific claim/begin/complete coordination is not the production worker path. The production worker invokes the same adapter operations directly without host timestamps. Existing deterministic M6B lease logic and tests remain unchanged. Delivery can repeat after expiry; exactly-once processor side effects, renewal, retry limits and queue fairness are not guaranteed.

Configure `UGSL_DATABASE_URL` only for explicit PostgreSQL runtime invocation. It is held as a secret and excluded from settings repr; ordinary imports, health/schema routes and offline tests do not connect. `UGSL_WORKER_LEASE_SECONDS` defaults to 300 (1–86400); `UGSL_WORKER_POLL_SECONDS` defaults to 1 (0.05–60). Connections have a 10-second connection timeout, are context-managed, commit only on success, roll back on exceptions and close after every operation. Migrations run explicitly, in filename order, under a transactional advisory lock, with checksummed applied-file tracking; reruns are safe, and a failed invocation rolls back schema changes and tracking together. Migration SQL is packaged with the distribution; importing modules never migrates.

```powershell
.\.venv\Scripts\python.exe -m ugsl_ai_coach.infrastructure.postgres.migrate
```

`worker/runtime.py` exposes `AnalysisWorker` and `run_postgres_worker(processor, settings=None)`. Callers must supply an `AnalysisProcessor.process(AnalysisWorkItem) -> StructuredAnalysisResult`. There is deliberately no fake production processor or standalone CLI pretending to resolve opaque media references. The bounded loop claims, begins, delegates and finalizes, and waits interruptibly between iterations including errors. SIGINT/SIGTERM request shutdown; an in-flight processor may finish before exit. Unexpected processor/infrastructure exceptions leave the lease recoverable and **never synthesize learner-analysis FAILED**. Only a valid FAILED result actually supplied by the processor is persisted. JSON operational logs include event, validated analysis ID and exception class; exception messages, tracebacks, credentials and opaque media references are excluded.

The real database suite is explicitly gated by `UGSL_TEST_DATABASE_URL` in the process environment (a dedicated test database with schema-creation permission). Each test creates and removes only a validated randomly named test schema. It covers migrations/reruns/failure, atomic rollback, concurrent idempotency/claims, SKIP LOCKED, database clock and expiry, stale generations, processing/resume, all terminal outcomes, immutable evidence, coaching and reconnect durability. Without a configured test database, these tests visibly skip; serializer/configuration/worker tests stay offline.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/postgres -m postgres
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pip check
```

PostgreSQL stores no video bytes, frames, landmarks, DTW matrices, screenshots or generated audio files. Media remains opaque references. Object storage/resolution, externally callable analysis APIs, authentication/authorization and deployment wiring remain later M6 work; existing mounted health and contract-schema routes are unchanged. No LLM, TTS, frontend or original-source-document rewrite is included.

### M6D secure backend API and private media integration

M6D selects the internal endpoint contract, bearer service authentication, safe error envelope and S3-compatible media integration as **new engineering decisions**. These choices are not attributed to the original product documents. The AI Coach is internal to the UgSL platform backend: the frontend does not call this service directly and must never receive the service token. The platform backend authenticates users and enforces learner/teacher/admin permissions and attempt ownership before calling the Coach. The Coach authenticates and authorizes the backend service only. Keep the service private and use encrypted transport; deployment/network enforcement remains later work.

| Protected operation | Required service scope | Successful response |
| --- | --- | --- |
| `POST /api/v1/analyses` | `analysis:submit` | 202 with analysis ID, attempt ID and actual state |
| `GET /api/v1/analyses/{analysis_id}` | `analysis:read` | 200 with actual state; complete validated M2 result when terminal |
| `GET /api/v1/analyses/{analysis_id}/feedback` | `feedback:read` | 200 with stored validated M5 feedback, or 202 typed PENDING |

POST accepts only the existing `AnalysisSubmission` fields. After authentication, authorization and namespace validation, it resolves both object identities with HEAD, then persists their canonical pinned references through existing M6B/M6C atomic acceptance. No job is accepted unless both identities resolve. Idempotency compares the exact pinned identities; an unchanged retry returns the same analysis, while changed content/version at either logical key returns 409. Duplicate canonical requests return the existing logical analysis; changed payload returns 409. GET never produces analysis or coaching, mutates records, or returns media references. Unknown analysis returns 404. Existing public health and contract-schema routes are unchanged.

Protected requests use `Authorization: Bearer <service-token>`. `UGSL_SERVICE_TOKEN` is required only when the protected runtime is invoked; health/schema/import paths do not require it. The static adapter requires 32–256 ASCII bearer characters (optional terminal padding), rejects trivial configuration and uses constant-time comparison. Generate at least 32 cryptographically random bytes; a syntax check cannot prove entropy. Typed `ServiceAuthenticator` and `ServiceAuthorizer` ports separate identity from scope enforcement and allow replacement. The default backend principal has all three scopes. Missing/invalid credentials return 401; authenticated principals lacking scope return 403. OpenAPI declares HTTP bearer security and records each scope with `x-required-scopes`; it does not advertise an OAuth login flow.

Errors use `{"error":{"code":"ANALYSIS_NOT_FOUND","message":"Analysis was not found.","retryable":false}}`. Stable codes distinguish authentication (401), authorization (403), validation/invalid reference (422), missing analysis (404), idempotency/identity conflicts (409), database unavailability (503), media failures and unexpected server errors (500). Media errors also have distinct typed internal classes for missing learner object, missing profile, unavailable storage, oversized/unsupported video and corrupt/incompatible profile. POST resolves metadata but does not download media; submission-time storage failures use the safe error envelope. Later worker failures remain recovery events, not manufactured learner outcomes. Error handlers and an exception boundary exclude upstream text/tracebacks from responses and prevent generic exceptions from escaping to ASGI-server traceback logging. Operational logs contain safe event codes, validated analysis IDs and exception classes; no tokens, Authorization headers, credentials, full object keys, bytes, learner landmarks or PII are logged.

`media/ports.py` defines private media retrieval. `infrastructure/object_store/s3.py` uses `boto3>=1.35,<2.0`, a fixed configured bucket, optional region/endpoint and the standard AWS/Boto credential provider chain. No custom access-key settings, public URLs, presigning, browser uploads or arbitrary downloader exist. Only safe ASCII object keys within distinct configured `learner-videos/` and `reference-profiles/` namespaces are accepted. Schemes/colons, absolute paths, backslashes, traversal segments, encoded delimiters, controls and cross-namespace references are rejected. Requests cannot choose a bucket, host or endpoint. The adapter retrieves server-side private objects only.

HEAD checks size and content type before GET. Video defaults to **50 MiB**; profile JSON defaults to **5 MiB**, both configurable and bounded. Video containers are limited to `.avi` with `video/x-msvideo` and `.mp4` with `video/mp4`; actual decoding must also pass the existing OpenCV M3 validation, so arbitrary codecs or malformed files are not promised support. Profiles require `.json` and `application/json`. Encoded objects are rejected. The persisted `ugsl-object-v1` reference is deterministic base64url-encoded typed JSON containing kind, namespace-validated key, non-null VersionId when available and strong expected ETag. It contains no bucket, endpoint or credentials and is never returned through API responses. Worker HEAD and GET both use the accepted VersionId, or the accepted ETag via If-Match; response identities and stream metadata/byte counts are verified. Missing historical versions or mismatches fail without retrying latest, and streams always close. Unpinned legacy work is rejected rather than silently rebound. Downloads use OS-managed unpredictable temporary directories and fixed internal filenames, with cleanup on success and exception. No downloaded learner video is stored in the repository or PostgreSQL.

References use the strict versioned `ugsl-reference-profile-v1` JSON contract. `build_reference_profile` creates the artifact offline from a trusted expert M3 extraction; `serialize_profile` publishes its JSON representation for the platform's private reference store. It retains only existing M4 wrist trajectory observations (including unavailable samples), extraction context, explicit hand correspondence/body region, reference/profile identity, extraction version, extractor identity and exact comparison policy. No reference video, raw/full landmarks, frames, pixels, new linguistic label or learner data are included. Expert review and immutable publication remain the platform's responsibility; creating an artifact does not establish linguistic validation of the provisional M4 thresholds.

Production extractor identity includes the pinned MediaPipe package version and SHA-256 hashes of both local hand/pose assets. Loaded profiles must match their requested object identity, model identity, schema/extraction version, normalization/sampling and comparison policy. Invalid JSON (including duplicate keys/non-finite values), invalid typed data and incompatibility fail explicitly. The small M4 `compare_precomputed_movement` entry point shares the original comparison body with `compare_movement`; DTW, eligibility, confidence, abstention, MOVEMENT-only policy and provisional similarity interpretation are unchanged.

`ProductionAnalysisProcessor` resolves the profile, downloads learner video temporarily, opens the existing `MediaPipeExtractor`, invokes unchanged M3 `extract_video`, and invokes the shared M4 comparison to return its existing M2 result. It never extracts the expert reference video for every attempt. Unexpected storage, extraction or runtime exceptions propagate to the M6C worker recovery boundary and never synthesize M2 FAILED. Insufficient observations still produce UNANALYZABLE under existing M4 rules; an actual M4 FAILED result remains a legitimate supplied result. Each extractor and media download is context-managed.

```powershell
# Explicit worker composition; requires DB, private store and local model configuration.
.\.venv\Scripts\python.exe -m ugsl_ai_coach.worker.processor
# Offline API/security/storage/profile/processor integration tests.
.\.venv\Scripts\python.exe -m pytest tests/secure_integration
```

Clients and configuration initialize only during explicit protected operations or worker composition; importing the app makes no PostgreSQL/S3 connection. No database schema change is needed and migration 001 is untouched. Both learner and expert-profile versions are bound before acceptance, including across retries and worker restarts. Versioned storage retrieves the historical version even after replacement; unversioned storage safely refuses changed ETags. Historical-object retention remains a separate platform policy; pinning does not guarantee that a deleted version stays available.

M5 remains a separate durable artifact. Feedback GET returns only an existing, source-grounded coaching record; no lazy generation occurs. **Durable automatic M5 generation/recovery remains a gap**, as do learner media/evidence privacy and retention policy, rate limiting, production monitoring/metrics/alerts, audit requirements, deployment/container/process supervision and deployed end-to-end verification. These remain M6E/M6F work. No learner login, frontend, upload workflow, public storage, notifications, WebSockets/SSE, broker, LLM/TTS or deployment provider is added.
## M6E operational hardening

M6E engineering decisions add reliable downstream delivery and technical
operations. They do not define learner policy, product analytics, product audit
requirements, or a deployment platform.

```text
terminal M2 result (COMPLETED / UNANALYZABLE / legitimate FAILED)
    -> durable PENDING coaching work
    -> fenced claim -> existing M5 grounding / deterministic provider / validation
    -> immutable CoachingRecord + COMPLETED coaching work
```

Apply the ordered migrations explicitly before starting M6E runtimes:

```powershell
.venv\Scripts\python.exe -m ugsl_ai_coach.infrastructure.postgres.migrate
.venv\Scripts\python.exe -m ugsl_ai_coach.worker.processor
.venv\Scripts\python.exe -m ugsl_ai_coach.worker.coaching
```

Migration `001` is unchanged. `002_operational_hardening.sql` adds
`coaching_work`, `service_rate_limits` and bounded `operational_worker_counts`.
An analysis-terminal trigger creates coaching work in the same transaction as
M6C finalization, including deferred integrity checks. A failure to schedule
rolls back the new terminal result and analysis-work completion. Coaching
itself runs in a later transaction; terminal analysis never implies feedback
is already available. `GET /feedback` retains its existing pending response.

The upgrade backfills terminal jobs without feedback as pending delivery. Jobs
with existing feedback are completed and preserve that feedback identity.
Checksum-tracked migration reruns do nothing. Historical evidence and feedback
are never updated or deleted by M6E.

New work reserves a stable `FB-<UUID>` string derived from the analysis ID using
PostgreSQL's built-in MD5/UUID representation with an M6E namespace. This is a
non-secret delivery identifier, not a cryptographic security primitive. It is
persisted before the first provider call and reused on every retry. Existing
M6A append remains supported: valid existing feedback is authoritative and
completes its work atomically, even if an older caller chose another feedback
ID. Workers reread committed feedback before calling the provider and never
replace that history or weaken conflicting-feedback checks.

Coaching states are only `PENDING`, `CLAIMED`, `COMPLETED`. PostgreSQL
`clock_timestamp()` controls claims, lease expiry and retry eligibility.
`FOR UPDATE SKIP LOCKED`, incrementing delivery generation and ownership checks
fence concurrent/stale workers. Expired claims are recoverable. Active claims
are excluded; completed work never reopens. A processing failure returns an
active claim to pending with a durable exponential delay of 5, 10, 20 ... up to
300 seconds by default. The only stored failure category is
`COACHING_PROCESSING_ERROR`; exception text is never persisted. Retry timing
has a bounded delay, not a product-defined maximum retry count. A database
failure leaves lease recovery available.

Validated feedback insertion and work completion share one transaction.
Database triggers also reconcile existing M6A append operations. Lost
acknowledgement is resolved by rereading committed feedback; no second provider
call is needed when committed truth can be established. Provider, database,
metrics and logging failures never create M2 `FAILED` or reopen terminal M2.
The production M5 provider remains deterministic; **no external LLM vendor is
configured or added**, and no new learner history/personalization context is
collected. Future providers can use the existing M5 boundary.

The coaching loop handles each job independently, waits a bounded interval
after every iteration, and honors SIGINT/SIGTERM without starting new processing
after shutdown is requested. Process supervision remains M6F.

Authenticated internal services share PostgreSQL minute-window rate limits:
60 submissions and 600 analysis/feedback reads per minute by default. These
are M6E operational capacity/security defaults, adjustable through `UGSL_`
settings, not learner/product policy. Buckets use the authenticated service
principal's SHA-256 digest and operation class; no learner, attempt, media,
request or IP identifier is used. Each trusted configured principal has at
most two rolling rows; windows reset those rows rather than creating request
history. The row lock serializes increment/check using database time and
saturates rejection counts. Unavailable PostgreSQL fails closed with safe 503.
Rejection precedes object resolution/CV, returns the existing error envelope
with `RATE_LIMITED`, `retryable=true`, HTTP 429 and integer `Retry-After`.

Each HTTP request gets a fresh random `X-Request-ID`; client-supplied values are
not trusted. Structured operational events correlate requests using that ID
and route templates, method, status and duration. Security events cover auth,
authorization, conflicts, rate limits and storage identity violations. They
exclude tokens, request bodies, object keys, findings, feedback, PII, client IP,
credentials and exception text/tracebacks. Log sink failure is isolated. These
events are technical security telemetry, **not a learner-activity/product audit
trail**; DEC-019/product audit requirements remain unresolved. Infrastructure
access-log redaction/configuration outside this application belongs to M6F.

`GET /metrics` requires existing service auth with `metrics:read` scope. The
static backend authenticator grants all defined service scopes; injected
authenticators can narrow them. This endpoint is not in the submit/read buckets
so monitoring remains available while those buckets are saturated. Metrics
include HTTP counts/latency/status, rate rejections, claims/reclaims, worker
success/failure/retry, PostgreSQL availability, queue depths, oldest eligible
work age, terminal M2 counts and committed coaching count. Labels are bounded
route templates, normalized HTTP methods/status, worker/outcome or terminal
status; never analysis/attempt/feedback/request IDs, object keys or tokens.

API HTTP counters are process-local and reset on restart; scrape each API
instance and aggregate as appropriate. Independent production workers write
best-effort bounded shared operational counts after processing/finalization,
so worker telemetry is visible from the API metrics endpoint. These counts
are operational observations, not exactly-once delivery evidence. Queue,
terminal-analysis and coaching-completion gauges query durable database truth
and remain meaningful after restart. A failed snapshot sets
`ugsl_postgres_up=0`, clears stale queue/count samples and still serves metrics.
Telemetry collection/recording failures do not change evidence or delivery.

`GET /api/v1/health` remains lightweight liveness. M6E adds unauthenticated
`GET /api/v1/health/ready`, returning only `ready` (200) or `not_ready` (503).
This is an M6E engineering contract suitable for infrastructure probes. It
validates API runtime auth/storage configuration and performs a simple
PostgreSQL query. It never creates an S3 client, downloads media, requires a
particular object or attempts worker model inference. It is not proof of full
end-to-end storage/CV/provider health.

Provider-neutral alerts live in `ops/prometheus/alerts.yml`, using the actual
implemented metric names for DB unavailability, persistent 5xx, worker failures,
stalled queues and rate-limit rejection volume. Thresholds are tunable M6E
engineering defaults. Prometheus scraping, rule validation with `promtool`,
alert delivery and supervision are M6F/operator responsibilities. No external
monitoring SaaS, broker or notification service is added.

Learner downloads use the service-owned `ugsl-ai-owned-media-v1` directory
under the runtime user's OS temp directory, marked ownership, unpredictable
`work-*` directories, neutral filenames and immediate context cleanup. Analysis
worker startup explicitly cleans marked abandoned directories older than
`UGSL_TEMP_MEDIA_MAX_AGE_SECONDS` (24 hours by default); import never cleans.
Active file locks preserve even old downloads. Cleanup only handles the known
flat service artifact set, rejects symlinks/reparse points and unexpected files,
and never recursively deletes arbitrary temp directories or S3 objects.
The runtime temp directory must remain private to its OS account.

This crash-residue age is a technical local-artifact decision. **Behavioral-data
retention, historical evidence/feedback retention, learner-media deletion and
exact product monitoring/audit policies remain open product/security decisions.**
M6E collects no clickstream, engagement, attention, IP history or device signals.

Real PostgreSQL tests are a release gate before commit. Set
`UGSL_TEST_DATABASE_URL` privately and run the full suite; each database test
uses a disposable owned schema. Without that setting, tests skip explicitly:
offline tests do not prove PostgreSQL transaction/concurrency behavior.

## M6F deployment package

The deployment target is Render, an M6F engineering choice. The reviewed Blueprint
prepares one API, one analysis worker, one coaching worker and managed PostgreSQL,
with external private S3-compatible storage. Automatic deployment is disabled.
See [the deployment and live verification runbook](ops/deployment.md) for runtime
commands, configuration, staged rollout, operator-only E2E, recovery and rollback.

The same Python 3.13 Docker image runs all three processes under a non-root user.
Exact M3 version-1 models are packaged and hash-verified after wheel installation.
GitHub Actions gates the complete suite on PostgreSQL 18, then wheel and image
checks. Local tests do not execute the live E2E tool or create cloud resources.

M6F deployment package ready for review. Live deployment and end-to-end verification
remain the final release gate; this repository preparation does not claim a deployed
or production-verified system.
