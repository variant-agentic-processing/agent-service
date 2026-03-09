# agent-service

Natural language query API for the [Variant Agentic Processing](https://github.com/variant-agentic-processing) platform. Accepts a plain English question, runs a Claude reasoning loop against the MCP variant server, and streams the answer back as SSE events.

## Overview

- **POST /query** — submit a natural language question; response is an SSE stream of `tool_call`, `tool_result`, `answer`, `error`, and `done` events
- **GET /health** — liveness probe, also reports how many MCP tools were discovered at startup

All routes are open within the VPN — no API key required. The VPN and Cloud Run internal ingress (`INGRESS_TRAFFIC_INTERNAL_ONLY`) are the security boundary.

## Prerequisites

- `mcp-variant-server` deployed and reachable (internal Cloud Run URL)
- Anthropic API key stored in Secret Manager as `anthropic-api-key` (or set `ANTHROPIC_API_KEY` in `.env`)
- VPN connected before working with the deployed service
- `genomic-pipeline` Artifact Registry repository exists

## Setup

```bash
cp .env.example .env
# Edit .env: set GCP_PROJECT, MCP_SERVER_URL, ANTHROPIC_API_KEY (for local dev)
```

Install dependencies:

```bash
poetry install
```

## Running locally

For local dev the service needs to reach the MCP server. Get an OIDC token and set it in `.env`:

```bash
gcloud auth print-identity-token --audiences=https://mcp-variant-server-fno64g2krq-uc.a.run.app
# Add to .env: MCP_IDENTITY_TOKEN=<token>
```

Then start the server:

```bash
poetry run uvicorn src.main:app --reload --port 8080
```

Ask a question via CLI:

```bash
poetry run poe ask -- --question "What pathogenic variants does HG002 have in BRCA2?"
```

Or use `agent-service.http` with the REST Client VS Code extension.

## Build

Builds the Docker image via Cloud Build and pushes to Artifact Registry:

```bash
poetry run poe build
```

## Deploy

First time only — log in and initialise the stack:

```bash
poetry run poe login
poetry run poe stack-init
```

Then deploy (and on all subsequent updates):

```bash
poetry run poe deploy
```

## Poe tasks

| Task | Description |
|------|-------------|
| `poetry run poe build` | Build and push Docker image via Cloud Build |
| `poetry run poe login` | Log into the Pulumi GCS backend |
| `poetry run poe stack-init` | Initialise the Pulumi stack (first time only) |
| `poetry run poe deploy` | Deploy or update the Cloud Run service via Pulumi |
| `poetry run poe ask` | Ask a question via CLI (pass `-- --question "..."`) |
| `poetry run poe test` | Run unit tests |
| `poetry run poe lint` | Run ruff linter |
| `poetry run poe logs` | Tail Cloud Run logs |

## Project structure

```
agent-service/
├── Dockerfile
├── pyproject.toml
├── agent-service.http       # REST Client test file (Phase 8 validation queries)
├── src/
│   ├── main.py              # FastAPI app + lifespan (tool discovery, Anthropic client init)
│   ├── agent.py             # Claude reasoning loop (max 10 iterations, SSE event stream)
│   ├── mcp_client.py        # MCP Streamable HTTP client + OIDC auth
│   └── secrets.py           # Secret Manager fetch with env var fallback
├── tests/
├── deploy/
│   ├── __main__.py          # Pulumi Cloud Run deploy
│   ├── Pulumi.yaml
│   └── Pulumi.dev.yaml
└── scripts/
    ├── ask.py               # poe ask — CLI query script
    ├── login.py             # poe login helper
    └── build_image.sh       # Cloud Build wrapper
```

## License

[CC BY-NC 4.0](LICENSE) — © 2025 Ryan Ratcliff. Free for non-commercial use with attribution. Commercial use requires prior written consent.
