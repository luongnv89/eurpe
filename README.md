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
[`frontend/README.md`](frontend/README.md) for details.

### Tests

```bash
pytest -q
```

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
