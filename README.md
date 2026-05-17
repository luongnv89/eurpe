# EURPE — EU Research Project Proposal Expert

EURPE is a fully-local, privacy-respecting AI assistant for drafting EU
research proposals (Horizon Europe, Horizon 2020, Digital Europe, CEF, and
related programmes). It indexes your past funded and rejected proposals plus
ESR notes locally, and uses a multi-agent RAG workflow to help proposal
coordinators draft new sections with explicit source-status labelling
(funded / rejected / ESR / unknown).

**Everything runs offline by default.** No cloud LLMs, no external embeddings,
no telemetry. The application is intentionally bound to `127.0.0.1` and the
configuration ships with `offline_mode: true`.

> Status: **pre-alpha**. This commit lands the workspace scaffold (Issue #1).
> Real ingestion (Docling), retrieval (Chroma + nomic-embed), and generation
> (LangGraph) land in subsequent issues.

## Repository layout

```
eu-research-projects/
├── prd.md                      # Product Requirements Document
├── tasks.md                    # Sprint task breakdown
├── pyproject.toml              # PEP 621 metadata + hatchling build
├── config.example.yaml         # Local config template (committed)
├── config.yaml                 # Real config (created by `eurpe smoke`, gitignored)
├── src/eurpe/
│   ├── __init__.py
│   ├── config.py               # Typed YAML config loader
│   ├── cli.py                  # `eurpe` Typer CLI (smoke, version)
│   ├── api/                    # FastAPI app (local /health endpoint)
│   ├── ingestion/              # Docling parsing + chunking (Issue #3)
│   ├── retrieval/              # Hybrid search over Chroma (later)
│   ├── generation/             # LangGraph multi-agent workflow (later)
│   └── export/                 # Markdown / DOCX exporters (later)
├── tests/                      # pytest suite (config + smoke)
├── frontend/                   # React 18 + Vite 5 + Tailwind 3 + shadcn/ui
└── data/                       # Local corpus + index (gitignored)
```

## Tech stack

| Layer            | Tool                                                           |
| ---------------- | -------------------------------------------------------------- |
| Orchestration    | [LangGraph](https://langchain-ai.github.io/langgraph/) (Python) |
| PDF parsing      | [Docling](https://github.com/DS4SD/docling)                    |
| Local LLM        | [Ollama](https://ollama.com/) (default: `llama3.1:8b`)         |
| Embeddings       | `nomic-embed-text` via Ollama                                  |
| Vector store     | [Chroma](https://www.trychroma.com/)                           |
| Backend          | [FastAPI](https://fastapi.tiangolo.com/) + [Typer](https://typer.tiangolo.com/) |
| Frontend         | React + Vite + Tailwind CSS + shadcn/ui                        |
| Storage          | Local filesystem only                                          |

## Quick start

### Backend

```bash
# 1. (recommended) create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. install the package in editable mode with dev extras
pip install -e ".[dev]"

# 3. verify the local setup with no network calls
eurpe smoke
```

The `eurpe smoke` command bootstraps `config.yaml` from
`config.example.yaml` if it is missing, creates the `data/corpus` and
`data/index` directories, and prints the resolved configuration. It exits
with code `0` on success and never attempts a network call.

To run the (currently minimal) local API:

```bash
uvicorn eurpe.api.main:app --host 127.0.0.1 --port 8765
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The dev server starts on `http://127.0.0.1:5173`. See
[`frontend/README.md`](frontend/README.md) for details, and
[`docs/accessibility.md`](docs/accessibility.md) for the v1
accessibility baseline (contrast audit, keyboard tab traces, and the
deferred-to-v1.2 list).

### Tests

The suite is split into two tiers by pytest marker so the inner-loop dev
cycle stays fast.

```bash
# Fast tests only — skips `e2e` and `docling` markers.
pytest -m 'not e2e and not docling' -q

# Equivalent convenience target.
make test

# Full end-to-end pipeline (ingest → retrieve → generate → audit) over
# every PDF in proposals/ plus tests/e2e/fixtures/. Single command, no
# prompts; honours `EURPE_E2E_OUTPUT_DIR` for artefact placement.
make e2e
# Or directly:
pytest tests/e2e/ -m e2e -v
```

**E2E corpus contract.** The E2E suite picks up every `*.pdf` it finds
in `proposals/` (developer-only — `.gitignore` excludes the directory
from version control) and `tests/e2e/fixtures/` (the synthetic
`sample_proposal.pdf` is generated on demand from `reportlab`). To E2E
a new proposal, drop the PDF into `proposals/` — no code change needed.
Per-proposal metadata can be supplied via a sibling `<stem>.yml` next
to the PDF; if absent, the suite synthesises sensible defaults
(`programme: horizon_europe`, `outcome: funded`, `proposal_title:
<stem>`).

**Running E2E against a real LLM (Ollama).** By default the E2E suite
uses a deterministic offline stub for the generation step so CI works
on a fresh checkout without Ollama installed. The stub emits clearly
labelled placeholder text — fine for structural assertions, but
unsuitable for evaluating real-proposal quality. To require a real
local LLM, start Ollama and set `EURPE_E2E_REQUIRE_LLM=1`:

```bash
# One-time setup.
ollama serve &
ollama pull llama3.1:8b   # or whichever model your config.yaml names

# Run the suite with the real-LLM gate enabled.
EURPE_E2E_REQUIRE_LLM=1 make e2e
# Or via the convenience target:
make e2e-real
```

When `EURPE_E2E_REQUIRE_LLM=1` is set and the deterministic stub is
hit (because Ollama is unreachable, the model is missing, or
`models.ollama_base_url` in `config.yaml` is wrong), the affected
case fails with a message naming the env var and the next steps. When
the env var is unset, the suite continues to accept the stub — that
preserves the CI-on-fresh-checkout contract.

**CI policy.** `.github/workflows/e2e.yml` runs the slow tier on every
push to `main`, nightly at 03:00 UTC, and on manual dispatch. All
other workflows should use `pytest -m 'not e2e'` so they stay
fast.

## Schema

Proposal and chunk metadata are typed Pydantic v2 models in
[`src/eurpe/schema/`](src/eurpe/schema/). Every chunk carries an explicit
`source_status` drawn from the closed set **`funded` / `rejected` /
`esr_note` / `unknown`**, and a validator guarantees that a chunk's status
can never drift from its parent proposal's `outcome`. See
[`tests/fixtures/metadata/`](tests/fixtures/metadata/) for one round-trip
example per status.

## Privacy guarantees

- `offline_mode: true` is the default in `config.example.yaml`; it must be
  preserved in `config.yaml` for confidential proposal work.
- The FastAPI server binds to `127.0.0.1` only (see
  `src/eurpe/api/main.py`).
- The Vite dev server binds to `127.0.0.1` only (see
  `frontend/vite.config.ts`).
- All proposal content is excluded from version control by `.gitignore`
  (`proposals/`, `corpus/`, `*.pdf`, `*.docx`, `*.pptx`, `*.xlsx`, etc.).

## License

Proprietary — internal Montimage tooling. See `pyproject.toml`.
