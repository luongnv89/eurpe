# EURPE — Claude Code project guide

EURPE is a **fully-local, privacy-respecting** assistant for drafting EU
research proposals. See @README.md for the full overview and tech stack.

**YOU MUST preserve the offline-first invariant.** This project handles
confidential proposal content. Do not add cloud-LLM calls, external
embeddings, telemetry, or non-`127.0.0.1` bindings to the default path
without an explicit issue authorising it.

## Hard rules

- **Never commit or push without an explicit user request.** Even after
  green tests.
- **Never put `Co-Authored-By: Claude` in commit messages.**
- **Always activate `.venv` before running any Python command.**
  `source .venv/bin/activate` — see @README.md Quick start.
- Default config (`config.example.yaml`) MUST keep `offline_mode: true`.
- The FastAPI app binds to `127.0.0.1` only. The Vite dev server binds
  to `127.0.0.1` only. Do not change either.
- Proposal content (`proposals/`, `corpus/`, `*.pdf`, `*.docx`, …) is
  gitignored. Do not stage it.

## Workflow preferences

- Use `gh` for all GitHub work (issues, PRs, releases). Never parse text
  output — always `--json` with explicit fields.
- Branch naming is type-prefixed via `/issue-resolver`
  (`fix/`, `feat/`, `refactor/`, …). See `.gitissue.yml` for the policy.
- One PR per issue. Use `Closes #N` in the PR body.
- For UI/frontend work: run the dev server (`cd frontend && npm run dev`)
  and exercise the feature in a browser before declaring done.

## Test commands

- `make test` — fast tier (skips `e2e` and `docling` markers). Default
  inner-loop command.
- `make e2e` — full E2E pipeline against every PDF in `proposals/` +
  `tests/e2e/fixtures/`. Uses the deterministic offline stub by default.
- `make e2e-real` — E2E with `EURPE_E2E_REQUIRE_LLM=1`. Requires
  `ollama serve` and the configured model pulled.
- `make test-all` — fast + docling + e2e. Pre-merge gate.
- Single test: `pytest tests/test_foo.py::test_bar -q`.

## Project layout

| Path | Purpose |
|------|---------|
| `src/eurpe/cli.py` | `eurpe` Typer CLI (`smoke`, `pilot`, …) |
| `src/eurpe/config.py` | Typed YAML config loader |
| `src/eurpe/api/` | FastAPI app (local `/health` endpoint) |
| `src/eurpe/ingestion/` | Docling parsing + chunking |
| `src/eurpe/retrieval/` | Hybrid search over Chroma |
| `src/eurpe/generation/` | LangGraph multi-agent workflow |
| `src/eurpe/generation/llm.py` | `LLMClient` interface + `make_llm_client` factory |
| `src/eurpe/schema/` | Pydantic v2 models (proposal, chunk, status) |
| `src/eurpe/pilot/` | MVP pilot validation orchestrator |
| `frontend/` | React 18 + Vite 5 + Tailwind 3 + shadcn/ui |
| `tests/e2e/` | Slow tier — gated by the `e2e` pytest marker |

## Schema invariant

Every chunk carries a `source_status` from the closed set
`funded` / `rejected` / `esr_note` / `unknown`. A validator guarantees a
chunk's status cannot drift from its parent proposal's `outcome`.
**Do not relax this validator.** See `tests/fixtures/metadata/` for
round-trip examples.

## Gotchas

- `eurpe smoke` bootstraps `config.yaml` from `config.example.yaml` if
  missing. Don't commit `config.yaml` — it's gitignored on purpose.
- The E2E suite picks up *every* `*.pdf` under `proposals/` (developer
  machines only). Per-PDF metadata via sibling `<stem>.yml`; defaults
  synthesised if absent.
- Pilot reports live under `release-notes/pilots/<release-tag>.md` and
  are the MVP release gate (Task 3.7 / #21).
- Frontend dev server: `http://127.0.0.1:5173`. Backend API:
  `http://127.0.0.1:8765`.

## Issue-driven workflow

This repo uses the gitissue skills with `.gitissue.yml`. Common flows:

- `/issue-creator` to file new work as structured GitHub issues.
- `/issue-resolver` to take an issue end-to-end (branch → code → PR).
- `/issue-triage`, `/issue-analysis`, `/issue-pr-review` for the rest.

Issue bodies capture **intent only** — no file paths, no implementation
hints. Resolver/analysis skills produce those fresh.

## Token Efficiency
- Never re-read files you just wrote or edited. You know the contents.
- Never re-run commands to "verify" unless the outcome was uncertain.
- Don't echo back large blocks of code or file contents unless asked.
- Batch related edits into single operations. Don't make 5 edits when 1 handles it.
- Skip confirmations like "I'll continue..." Just do it.
- If a task needs 1 tool call, don't use 3. Plan before acting.
- Do not summarize what you just did unless the result is ambiguous or you need additional input.
