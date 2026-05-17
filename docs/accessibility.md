# Accessibility — v1 baseline

EURPE ships a "best-effort accessibility baseline" in v1 (Task 3.6 /
issue #20). Full WCAG 2.1 AA validation, axe-core automation, and
screen-reader regression are explicitly deferred to v1.2 (Task 3.6 in
the PRD's *v1.2 — UI refactor* section).

This document is the durable handoff for the v1.2 refactor team: it
records what v1 covers (with file references and verification steps),
the numerical contrast measurements behind the design tokens, and the
gaps that remain.

---

## What v1 covers

### AC #1 — Keyboard navigability with visible focus states

- **Skip-to-main-content link** is rendered first on every view
  (`frontend/src/App.tsx::SkipLink`). It is `sr-only` until focused, at
  which point it appears as a top-left chip with a visible ring.
- **Visible focus ring** is built into the shadcn primitives the app
  uses: every `Button`, `Input`, `Textarea`, and `SelectTrigger` carries
  `focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2`
  (`frontend/src/components/ui/button.tsx`, `input.tsx`, `textarea.tsx`,
  `select.tsx`). The `lessons-learned` raw `<input type="checkbox">` in
  `DraftingWorkspace.tsx` was given the same ring utilities so it does
  not drop out of the tab order's visual feedback.
- **No keyboard traps** on the primary flows. The drag-and-drop area in
  `IngestWizard.tsx` is `role="presentation"`; the visible **Choose
  file** Button is the keyboard-accessible entry point and is in the
  natural tab order.
- **Manual tab trace** (recorded 2026-05-17):

  *Home view*
  1. Skip link (hidden, first focus)
  2. **Ingest a proposal** button
  3. **Draft a section** button
  4. **Browse corpus** button (skipped — `aria-disabled`)

  *Ingest flow (upload step)*
  1. Skip link
  2. **← Home** button (TopBar)
  3. **Choose file** button
  4. (after parse) form fields in source order, then **Cancel**, then
     **Confirm and index**.

  *Drafting flow*
  1. Skip link
  2. **← Home** button (TopBar)
  3. Section type select → drafting profile select → context tabs →
     free-text or structured fields → user intent textarea → programme
     filter → top-k → critic iterations → lessons-learned checkbox →
     **Reset** → **Generate draft**
  4. (after generate) **Accept draft** → **Refine** → focusable
     citation / iteration entries inside the draft region.

### AC #2 — Labels on inputs, buttons, and generated report regions

- **Form inputs.** Every interactive control in `ConfirmationForm.tsx`
  and `DraftingWorkspace.tsx` is paired with a `<Label htmlFor>` (the
  Radix-backed `Label` primitive). Required controls also carry
  `aria-required="true"`. When client-side validation fails, the same
  controls flip to `aria-invalid="true"`; the per-error list is rendered
  inside a `role="alert"` Alert so assistive technology announces the
  message at the moment it appears.
- **Buttons.** Icon-only or icon-leading buttons have explicit
  `aria-label` text (`Accept this draft and stop the critic loop`,
  `Run one more critic loop iteration`, `Return to the EURPE home
  view`). Decorative `lucide-react` icons inside buttons and alerts use
  `aria-hidden="true"` so screen readers do not double-announce the
  visible text label.
- **Generated report region.** `DraftPreview` is wrapped in a
  `role="region"` with `aria-label="Generated <Section> draft"`. The
  `<pre>` block that contains the LLM output carries its own
  `aria-label="<Section> draft text"`. The home-page top bar is a
  `<header>`, and each feature view wraps its content in a `<section>`
  with `aria-labelledby` so the document has a proper landmark
  structure.
- **Status / busy state.** Long-running operations announce themselves
  via `role="status"` + `aria-live="polite"` (the PDF parsing spinner,
  the "Generating draft" hidden status line, the success card after
  indexing, and the iteration counter). The form regions that own those
  operations also carry `aria-busy` while a request is in flight so
  assistive technology knows the inputs are temporarily volatile.

### AC #3 — Contrast tokens meet readable targets

Audit performed against the white light-mode background (`--background
0 0% 100%`) using the WCAG 2.1 relative-luminance formula. Targets are
4.5:1 for normal body text, 3:1 for non-text UI components and large
text (≥18 pt or 14 pt bold).

| Token (light) | Old value | Old ratio | New value | New ratio | Target | Result |
|---|---|---|---|---|---|---|
| `--foreground` on `--background` | `222.2 84% 4.9%` | 19.99:1 | unchanged | 19.99:1 | 4.5 | PASS (AAA) |
| `--muted-foreground` on `--background` | `215.4 16.3% 46.9%` | 4.75:1 | `215.4 16.3% 38%` | **6.58:1** | 4.5 | PASS (AAA) |
| `--primary-foreground` on `--primary` | `210 40% 98%` on `222.2 47.4% 11.2%` | 17.04:1 | unchanged | 17.04:1 | 4.5 | PASS (AAA) |
| `--destructive-foreground` on `--destructive` | `210 40% 98%` on `0 84.2% 60.2%` | 3.59:1 | `0 72% 45%` | **5.55:1** | 4.5 | PASS (AA) |
| `--ring` on `--background` | `222.2 84% 4.9%` | 19.99:1 | unchanged | 19.99:1 | 3 | PASS |
| `--border` on `--background` | `214.3 31.8% 91.4%` | ~1.27:1 | unchanged | ~1.27:1 | 3* | NOTE (see below) |

*The border token sits below the 3:1 non-text contrast target. That is
the stock shadcn behaviour, justified for v1 because every interactive
border (input, button outline) is paired with a 4.5:1 label/value and a
strong `focus-visible` ring; the border itself is only an aesthetic
hint, not the affordance. v1.2 should re-evaluate alongside the full UI
refactor.

Status badges in `DraftPreview.tsx` use stock Tailwind 100/800 palette
pairs, all of which pass AA:

| Badge | Pair | Ratio |
|---|---|---|
| FUNDED | `text-green-800` on `bg-green-100` | 6.49:1 |
| REJECTED | `text-red-800` on `bg-red-100` | 6.80:1 |
| ESR ADVISORY | `text-blue-800` on `bg-blue-100` | 7.15:1 |
| UNKNOWN | `text-slate-800` on `bg-slate-100` | 13.35:1 |

Dark-mode tokens already cleared AA against `--background 222.2 84%
4.9%` (foreground 19.09:1, muted-foreground 7.80:1) and were left
unchanged.

### AC #4 — This document

The remaining-gaps section below is the v1.2 hand-off contract.

---

## Resizable text behaviour

EURPE relies on Tailwind's `rem`-based type scale (the `text-*` and
`leading-*` utilities) and on the user-agent default root font size.
That means a user who bumps their browser default to 24 px (200%) sees
the entire UI scale proportionally — there are no `px`-locked font
sizes in the application source. Container widths use `max-w-3xl`,
`max-w-5xl` (also `rem`-based), so the content reflows correctly inside
a 320 px viewport at 200% zoom. We do not declare `viewport`
`user-scalable=no` and do not set fixed line heights; the WCAG 1.4.4
"Resize text up to 200%" technique is therefore satisfied by default.

---

## Verification steps a reviewer can run manually

The v1 baseline has no automated a11y suite; reviewers should walk
through the following checks before approving a UI-touching PR:

1. **Build.** `cd frontend && npm run build`. Must succeed; the build
   exercises every component and `tsc --noEmit` catches missing
   `aria-*` typings.
2. **Tab trace.** With the dev server running, press `Tab` from a
   freshly loaded view and confirm the tab order matches the trace
   listed under AC #1 above. The skip link must be the first focusable
   element and must become visible on focus.
3. **Visible focus.** Each control must show a 2 px ring on focus.
   shadcn primitives ship this; the only hand-rolled control is the
   `lessons-learned` checkbox in `DraftingWorkspace.tsx`.
4. **Error announcement.** Submit the drafting form with an empty
   `user_intent`. The Alert must appear, the textarea must show
   `aria-invalid="true"` in DevTools, and a screen reader (VoiceOver
   `Cmd+F5`, NVDA, Orca) must announce the error list.
5. **Generated report.** After a draft renders, a screen reader's
   landmarks list must show *Generated &lt;Section&gt; draft* as a
   region.
6. **Contrast spot-check.** Open the deployed light view in a contrast
   tool of your choice (browser DevTools, Stark, axe DevTools) and
   verify body text + muted helper text clear 4.5:1.

---

## Deferred to v1.2

The following work is explicitly out of scope for v1 and tracked for
the v1.2 UI refactor (PRD: *v1.2 — Full WCAG 2.1 AA validation and
hardening*):

- **Automated a11y suite.** Wire `axe-core` (or `@axe-core/playwright`)
  into a frontend e2e harness. No test framework is configured for the
  React side today; the v1.2 refactor should adopt one (vitest +
  testing-library at minimum) and add an axe sweep to every primary
  flow.
- **Screen-reader regression.** Record an explicit VoiceOver / NVDA /
  Orca pass for the home, ingest, and drafting flows and capture the
  transcript as a test artefact.
- **`prefers-reduced-motion`.** The drafting and ingesting spinners,
  status alerts, and the `accordion-down/up` Tailwind animations should
  honour `@media (prefers-reduced-motion: reduce)` (snap to final
  state, no spin).
- **High-contrast / forced-colors mode.** Audit the slate palette under
  Windows High Contrast and macOS Increase Contrast. Border tokens
  (currently below 3:1 — see AC #3 note) need a forced-colors
  alternative.
- **Colour-blind audit.** Verify the FUNDED / REJECTED / ESR ADVISORY
  badges are still distinguishable without colour (the glyph and the
  text label already convey the meaning, but the badges should be
  spot-checked under deuteranopia and protanopia simulators).
- **Focus management.** Move keyboard focus into the success card after
  a successful ingestion, and into the generated draft region after a
  successful generation. Currently the `role="status" aria-live`
  attributes announce the change but focus stays on the trigger button.
- **Custom widget audit.** The Radix `Select`, `Tabs`, and `Label`
  primitives ship with ARIA roles, but the v1.2 effort should record a
  formal AA conformance statement for each.
- **Dropzone keyboard parity.** v1 ships the dropzone as a
  pointer-only enhancement next to a keyboard-accessible
  **Choose file** button. v1.2 should consider promoting the dropzone
  itself to a `role="button"` with `tabIndex={0}` and `Enter`/`Space`
  handlers if user research shows the dual-path UX is confusing.
- **Form-error focus.** Move keyboard focus to the first invalid field
  on submit. Today the Alert appears in-page and the user has to find
  the bad field manually.
- **i18n & `lang` per-region.** `<html lang="en">` is set globally;
  v1.2 should revisit if any UI strings ever switch locale at runtime
  (e.g., a French call summary embedded inside a generated draft).

---

## Source-of-truth references

- Acceptance criteria — GitHub issue #20 (Task 3.6).
- Programme commitments — `prd.md`, sections *Accessibility* and
  *Release Planning v1.2*.
- Slate palette and CSS variables — `frontend/src/index.css`.
- Primary flows touched by this baseline — `frontend/src/App.tsx`,
  `frontend/src/features/ingest/{IngestWizard,ConfirmationForm}.tsx`,
  `frontend/src/features/drafting/{DraftingWorkspace,DraftPreview}.tsx`.
