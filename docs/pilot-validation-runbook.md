# MVP pilot validation runbook (Task 3.7 / issue #21)

This runbook is the operator-facing procedure for the MVP go/no-go
gate described in `prd.md` (line 173: "Coordinators can generate
real proposal sections with ≥4/5 satisfaction on at least one
active call, with citations that clearly label funded/rejected/ESR
source status"). It pairs with:

- the orchestrator at `src/eurpe/pilot/` (CLI: `eurpe pilot run`),
- the report template at `docs/pilot-report-template.md`, and
- the committed sample run at
  `release-notes/pilots/v1.0-pilot-smoke.md`.

There are two flavours of pilot:

1. **Smoke pilot** — fully offline, deterministic stubs. Validates
   that the plumbing (smoke probe, indexing, retrieval, generation,
   audit, benchmark) holds under the offline-by-default contract.
   Default mode of `eurpe pilot run`. Renders the verdict
   `CONDITIONAL` by construction: a smoke run is *not* a release
   sign-off.
2. **Coordinator pilot** — Ollama-backed run with real human
   ratings collected per section. The release gate. Renders the
   verdict `GO` only when every section's mean satisfaction is ≥ 4
   AND the audit / smoke invariants hold.

## Prerequisites

- A clean repo: `make test` is green.
- Python venv activated (`source .venv/bin/activate`).
- `eurpe smoke` exits 0 (proves the network-policy gate and the
  workspace dirs are healthy).
- For the coordinator pilot only: Ollama installed and the
  configured model pulled (`ollama pull llama3.1:8b` or whatever
  `config.yaml` names).

## Step 1 — Run the smoke pilot

The smoke pilot is the first gate. Run it on a fresh checkout
before opening the coordinator session — if it fails, the
coordinator pilot will fail too (and waste your coordinator's
time).

```bash
eurpe pilot run \
  --output-dir release-notes/pilots/<release-tag>-smoke \
  --call-id HORIZON-CL5-2024-D3-02 \
  --proposal-title "MVP Pilot Synthetic Corpus" \
  --notes "Pre-release smoke; runtime: deterministic"
```

Inspect the stdout summary. The expected pattern for a clean smoke
run is:

```
Pilot mode      : smoke
Sections        : 3
Smoke probe     : PASS (exit 0)
Audit           : 0/3 drafts passed     ← expected; see below
Citation issues : 3                     ← all `placeholder_text`
Verdict         : CONDITIONAL
```

The 0/3 audit pass count and the 3 citation issues are **expected**
under the deterministic stub LLM: the stub emits a verbatim
placeholder sentence that the citation audit flags as
`placeholder_text` — by design, so a release never ships stub text.
The pilot's verdict logic tolerates this *one* code in smoke mode;
any other code (e.g., `missing_status`, `marker_without_citation`)
flips the verdict to `NO_GO`. If you see anything but
`placeholder_text`, fix the underlying issue before continuing.

If the smoke probe FAILs (exit code 1), the network-policy gate
allowed a TEST-NET probe — check your `network_allowlist` for an
over-broad entry. This is release-blocking.

## Step 2 — Run the coordinator pilot

Once the smoke pilot is CONDITIONAL, schedule a coordinator session
and run the real-LLM pilot.

```bash
# In one terminal: Ollama daemon must be up.
ollama serve &

# In another: run the coordinator pilot.
eurpe pilot run \
  --output-dir release-notes/pilots/<release-tag> \
  --mode coordinator \
  --runtime ollama \
  --call-id HORIZON-CL5-2024-D3-02 \
  --proposal-title "<real proposal title>" \
  --section-type methodology \
  --section-type impact \
  --section-type implementation \
  --notes "Coordinator <coord-a>, M1 Air, llama3.1:8b"
```

The coordinator pilot:

- Uses the real Ollama LLM and embedder.
- Produces per-section JSON + Markdown drafts under
  `--output-dir`.
- Records the **same** smoke + audit + benchmark fields as the
  smoke pilot.
- Renders `CONDITIONAL` until the coordinator rates every section
  (see Step 3).

## Step 3 — Capture coordinator ratings

The coordinator opens each section's rendered Markdown
(`<output-dir>/<section>.md`), reads the draft + the citation
table, and records:

- **Satisfaction:** 1–5 Likert rating. The PRD floor is 4.
- **Time saved:** approximate minutes vs. drafting the section
  manually. AC2 of issue #21.
- **Notes:** any specific issues (content-safe — no proposal
  content; the runbook's privacy contract is the same as the
  analytics events').

For each rating, run:

```bash
eurpe pilot rate \
  release-notes/pilots/<release-tag>/pilot-report.json \
  -s methodology \
  --coordinator-id coord-a \
  --rating 4 \
  --time-saved 45 \
  --notes "methodology section needed minor tightening"
```

The `rate` subcommand atomically rewrites the report JSON and
re-computes the verdict. After every section has at least one
rating ≥ 4, the verdict flips to `GO`.

If the verdict stays `NO_GO`, look at the citation issues and the
smoke probe in the rendered Markdown. The release is blocked until
they clear.

## Step 4 — Render the final Markdown report

Once the verdict is `GO` (or `NO_GO` with the release blockers
documented):

```bash
# Re-render the Markdown next to the JSON.
eurpe pilot run \
  --output-markdown release-notes/pilots/<release-tag>/REPORT.md \
  --output-json release-notes/pilots/<release-tag>/pilot-report.json \
  --mode coordinator --runtime ollama \
  --call-id HORIZON-CL5-2024-D3-02 \
  --proposal-title "<real proposal title>"
```

Or — simpler — open the JSON, copy it into a fresh
`docs/pilot-report-template.md`, and fill in the free-text fields
that the JSON does not carry (lessons learned, follow-up actions).

Commit the result under
`release-notes/pilots/<release-tag>.md`, attach to the release
tag.

## Step 5 — Decide

The release gate reads the report's `verdict` field:

- `GO` → cut the release.
- `CONDITIONAL` → schedule a coordinator pilot OR re-run after
  fixing the missing rating.
- `NO_GO` → fix the named issue, re-run the pilot from Step 1.

## What the pilot does NOT cover

- **Real PDF parsing performance.** The runner uses a synthetic
  in-memory corpus so it runs on a fresh checkout with no real
  proposals. For PDF parsing benchmarks, run `eurpe benchmark
  indexing --corpus proposals/` separately.
- **Multi-coordinator agreement.** v1 records one rating per
  coordinator per section; if you want N coordinators per section,
  run `eurpe pilot rate` N times with distinct
  `--coordinator-id` values.
- **Cross-call generalisation.** The pilot exercises one call. v1.1
  introduces multi-call evaluation per the PRD's
  *Post-MVP v1.1* section.
