# Release Audit Template — Citation Fidelity and Source Labelling

This template is the **human-judgement half** of the Task 3.4 release
gate. The automated half lives in
`src/eurpe/generation/audit_harness.py` (CLI:
`eurpe generate audit-release <directory>`), which fails the build
when any citation lacks a `funded`, `rejected`, `esr_note`, or
`unknown` status label. The two halves together satisfy issue #18's
acceptance criteria and the PRD success metrics for
**v1 citation fidelity** (≥ 95% of generated claims trace to real
past proposals) and **source labeling accuracy** (100% of citations
visibly show source status).

The template scaffolds an audit of **at least 20 generated sections
per release** (PRD line 21), with one checklist row per section.
Copy this file into `release-notes/audits/<release-tag>.md` before
cutting the release, fill in the verdict columns, and attach it to
the release tag for the regulator-style traceability the PRD demands.

## How to use

1. Generate the release sample with
   `eurpe generate audit-release <run-dir> --sample-size 20
   --output-markdown <release-audit>.md --output-json
   <release-audit>.json`. Use `--seed N` if you want to pin the
   subset across re-runs.
2. Copy the generated Markdown summary into the **Automated audit**
   section below. The harness fails fast on missing-label errors;
   if the run exited non-zero, fix the source draft (or the
   retrieval / generation pipeline) and re-run before continuing.
3. For each row in the **Manual checklist** below, open the cited
   PDF (or the per-row chunk in the index) and answer:
   - **Citation present?** — does the rendered draft cite the
     claim? (yes/no)
   - **Source matches?** — does the cited chunk actually support
     the claim in the draft? (yes/no — this is the v1 citation
     fidelity invariant)
   - **Status correct?** — does the visible badge
     (`✓ FUNDED`, `✗ REJECTED`, `ⓘ ESR ADVISORY`, `? UNKNOWN`)
     match the source's actual outcome? (yes/no — this is the
     source labelling invariant)
   - **Notes** — anything else that the next release should fix
     (e.g., "page number was off by one", "ESR note was framed
     as ground truth in the prose").
4. The **release gate** fails when any row reports `no` for any of
   the three yes/no columns. Fix the underlying issue before
   shipping; do not mark `no` rows as `n/a`.

## Automated audit

Paste the output of `eurpe generate audit-release` below:

```
<verdict will go here — paste the rendered Markdown report>
```

## Manual checklist (20 rows)

The harness output ranks drafts by sorted filename. Walk the table
top-to-bottom; the row numbers below align with the `Per-draft
Results` block in the automated section above.

| #  | Draft path | Section | Citation present? | Source matches? | Status correct? | Notes |
|----|------------|---------|--------------------|------------------|------------------|-------|
| 1  |            |         |                    |                  |                  |       |
| 2  |            |         |                    |                  |                  |       |
| 3  |            |         |                    |                  |                  |       |
| 4  |            |         |                    |                  |                  |       |
| 5  |            |         |                    |                  |                  |       |
| 6  |            |         |                    |                  |                  |       |
| 7  |            |         |                    |                  |                  |       |
| 8  |            |         |                    |                  |                  |       |
| 9  |            |         |                    |                  |                  |       |
| 10 |            |         |                    |                  |                  |       |
| 11 |            |         |                    |                  |                  |       |
| 12 |            |         |                    |                  |                  |       |
| 13 |            |         |                    |                  |                  |       |
| 14 |            |         |                    |                  |                  |       |
| 15 |            |         |                    |                  |                  |       |
| 16 |            |         |                    |                  |                  |       |
| 17 |            |         |                    |                  |                  |       |
| 18 |            |         |                    |                  |                  |       |
| 19 |            |         |                    |                  |                  |       |
| 20 |            |         |                    |                  |                  |       |

If the release samples more than 20 sections, append additional
rows below the table — keep the row-number column contiguous so
the audit cross-references in release notes remain stable.

## Verdict

- **Automated audit exit code:** `0` / non-zero
- **Manual rows with any `no` answer:** count: `___`
- **Release-blocking findings (rows that must fix before ship):**
  - …

**Final verdict:** `PASS` / `FAIL`

Signed: `<reviewer name>`
Date: `<YYYY-MM-DD>`
Release tag: `<v…>`
