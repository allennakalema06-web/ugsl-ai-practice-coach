# UgSL AI Practice Coach

The Uganda Sign Language (UgSL) AI Practice Coach will eventually compare learner practice performances with expert-validated references and provide evidence-based feedback.

## Current status and scope

Milestone 1 provides a standalone Python service foundation: FastAPI application factory, versioned process health endpoint, typed environment settings, standard-library JSON application logging, and automated tests. **The current service does not analyze UgSL signs.** No AI, video processing, storage, authentication, or frontend is implemented.

## Architecture principles

The model is a component of the AI Coach, not the AI Coach itself. Future objective CV and comparison stages will produce structured findings; a later LLM layer may explain those findings and must not independently judge sign correctness from raw video. Insufficient confidence must lead to abstention. Previous learner attempts will eventually be immutable historical records.

Requests currently flow through the FastAPI app in `src/ugsl_ai_coach/main.py` to the health router under `/api/v1`. Settings are loaded when the application is created and attached to that app. Logging is configured at startup, with startup/shutdown messages and DEBUG health messages. Application logs are JSON; Uvicorn retains its own server/access logging. `core/` contains configuration and logging, while `api/routes/` contains HTTP routes. Future domain, pipeline, comparison, coaching, and adapter modules will be added when needed.

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

Later milestones will address video validation/frame extraction, landmarks and normalization, trajectory representations and expert-reference comparison, confidence and abstention, structured findings and coaching explanations, and persistence/backend/frontend integration. These are future scope, not capabilities delivered by M1.
