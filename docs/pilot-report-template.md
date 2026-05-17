# MVP Pilot Validation Report — `<release-tag>`

This template is the **human-judgement half** of the Task 3.7 / issue
#21 release gate. The automated half is produced by `eurpe pilot
run` (see `docs/pilot-validation-runbook.md` for the procedure).

Copy this file into `release-notes/pilots/<release-tag>.md` before
cutting the release. Paste the automated artefacts into the
*Automated reports* section below; fill in the *Coordinator
narrative* section by hand.

## Header

- **Release tag:** `<v1.0.x>`
- **Pilot mode:** `coordinator` (or `smoke` for a pre-release dry
  run)
- **Active call:** `<HORIZON-CL5-2024-D3-02>` _AC1: ≥1 real call._
- **Proposal:** `<title of the proposal indexed for evidence>`
- **Coordinator:** `<coord-a>` _anonymous identifier; do not write a
  real name._
- **Runtime:** `ollama` (or `deterministic` for smoke)
- **Date (UTC):** `<YYYY-MM-DD>`

## AC1 — Sections drafted

_The pilot exercises **at least three** generated section drafts._

| # | Section | Citations | Audit | Draft artefact |
|---|---------|-----------|-------|----------------|
| 1 |         |           |       |                |
| 2 |         |           |       |                |
| 3 |         |           |       |                |

## AC2 — Coordinator ratings

_Every section is rated by the coordinator, with time-saved vs.
manual drafting._

| Section | Coordinator | Rating (1–5) | Time saved (min) | Notes |
|---------|-------------|--------------|------------------|-------|
|         |             |              |                  |       |
|         |             |              |                  |       |
|         |             |              |                  |       |

The PRD success criterion (`prd.md` line 173) is **mean ≥ 4/5
satisfaction on every section**. A row with rating < 4 is
release-blocking.

## AC3 — Required pilot fields

### Satisfaction

Aggregate from the table above: mean = `<x.y> / 5`. Release floor:
4.0.

### Citation issues

_Paste the citation-issues table from the `pilot-report.json`'s
`citation_issues` array (or `eurpe pilot run --output-markdown`
output). A clean coordinator pilot reports **zero** issues._

| Section | Code | Message | Draft |
|---------|------|---------|-------|
|         |      |         |       |

### Performance

_Paste the performance block from the rendered Markdown. Sanity
checks against PRD targets:_

- Indexing: `<chunks/sec>` (PRD target: 40 proposals in < 2 h on
  M1).
- Retrieval: avg `<x>` ms / p95 `<y>` ms (PRD target: < 2 s).
- Generation: `<z>` ms (PRD target: < 2 min on M1).

### Network isolation smoke

- Verdict: **PASS** / **FAIL** (exit code `<n>`)
- Detail: `<TEST-NET probe denied as expected | misconfigured allowlist…>`

### Go / No-Go recommendation

**`GO`** / **`NO_GO`** / **`CONDITIONAL`**

If `CONDITIONAL`, list the missing prerequisites.
If `NO_GO`, list the release-blocking findings and the planned fix
in the *Follow-up actions* below.

## Coordinator narrative

_Free-form notes the JSON cannot carry. Keep content-safe — no
proposal content, no client names. Focus on **what the tool got
right** and **what slowed the coordinator down** so the v1.1
roadmap can act on it._

### What worked

- …

### What slowed me down

- …

### Lessons learned for v1.1

- …

## Follow-up actions

- [ ] …

## Signatures

| Role | Name | Date |
|------|------|------|
| Coordinator |   |   |
| Release manager |   |   |

## Attached artefacts

- `pilot-report.json` — full structured run.
- `<section>.json`, `<section>.md` — per-section drafts.
- `index/` — Chroma collection used by the run (for re-audit).
