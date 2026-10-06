# UgSL AI Practice Coach

The Uganda Sign Language (UgSL) AI Practice Coach will eventually compare learner practice performances with expert-validated references and provide evidence-based feedback.

## Current status and scope

Milestone 1 provides a standalone Python service foundation: FastAPI application factory, versioned process health endpoint, typed environment settings, standard-library JSON application logging, and automated tests. **The current service does not analyze UgSL signs.** No AI, video processing, storage, authentication, or frontend is implemented.

Milestone 2 adds Structured Findings Contract v1: validated Pydantic domain models and a generated JSON Schema endpoint. It defines what future analysis components may report; it produces no analysis results or feedback.

## Architecture principles

The model is a component of the AI Coach, not the AI Coach itself. Future objective CV and comparison stages will produce structured findings; a later LLM layer may explain those findings and must not independently judge sign correctness from raw video. Insufficient confidence must lead to abstention. Previous learner attempts will eventually be immutable historical records.

Requests currently flow through the FastAPI app in `src/ugsl_ai_coach/main.py` to health and contract routers under `/api/v1`. Settings are loaded when the application is created and attached to that app. Logging is configured at startup, with startup/shutdown messages and DEBUG health messages. Application logs are JSON; Uvicorn retains its own server/access logging. `core/` contains configuration and logging, `api/routes/` contains HTTP routes, and `domain/analysis.py` defines the authoritative contract. Future pipeline, comparison, coaching, and adapter modules will be added when needed.

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

Tests use an in-process HTTP client and controlled settings. They need no external network, API keys, webcam, or database. Development dependencies are pytest and HTTPX (required by FastAPI's test client); runtime dependencies are FastAPI, pydantic-settings, and Uvicorn. No separate lint/format tool is introduced for M1.

## Future milestones

Later milestones will address video validation/frame extraction, landmarks and normalization, trajectory representations and expert-reference comparison, confidence and abstention, coaching explanations grounded in this contract, and persistence/backend/frontend integration. These remain future scope; M2 only defines and validates the structured evidence boundary.
