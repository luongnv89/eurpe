# Development Tasks - EURPE

> Generated from: [prd.md](./prd.md)
> Generated on: 2026-05-14

## Overview

### Development Phases
- **POC**: Prove the core local proposal-section drafting loop using a small indexed corpus, source-aware retrieval, and visible citation labels.
- **MVP**: Complete v1 ingestion, call/topic intake, drafting profiles, React/Vite/Tailwind/shadcn UI, offline/privacy controls, and Markdown/DOCX section export.
- **Post-MVP**: Add scoring rubric profiles, evaluator simulation, search/browse, human-in-the-loop controls, compliance checks, DGX deployment, and multi-user expansion.

### Sprint Overview

| Sprint | Phase | Focus | Task Count |
|--------|-------|-------|------------|
| 1 | POC | Local corpus ingestion, retrieval, and first section generation loop | 7 |
| 2 | MVP Foundation | Drafting profiles, call/topic intake, ingestion UI, offline audit controls | 7 |
| 3 | MVP Completion | React/Vite workflow, critic loop, export, benchmarks, pilot validation | 7 |
| 4 | Post-MVP v1.1 | Rubric profiles, evaluator simulation, interactive search, human-in-loop | 7 |
| 5 | Future v1.2/v2 | Compliance, wiki synthesis, full exports, DGX, accessibility, multi-user | 7 |

## Sprint 1: Proof of Concept (POC)

### Task 1.1: Scaffold the local EURPE application workspace

**Description**: Create the initial local application structure for EURPE, including configuration loading, local storage paths, a React/Vite frontend workspace, and a simple command or script entry point for running ingestion and generation workflows. This establishes the foundation for fully local execution described in the PRD.

**Acceptance Criteria**:
- [ ] Repository folder contains clear frontend, app/API, ingestion, retrieval, generation, export, and tests directories or modules.
- [ ] Frontend workspace starts with React, Vite, Tailwind CSS, and shadcn/ui configured.
- [ ] A local configuration file defines corpus path, index path, model runtime settings, and offline mode default.
- [ ] A smoke command runs without requiring any network connection.

**Dependencies**: None

**Effort**: 1-2 days (S)

**PRD Reference**: Technical Specifications; Security; Release Planning MVP

---

### Task 1.2: Define proposal metadata and source-status schema

**Description**: Implement the core metadata model needed to distinguish programme, call, topic, outcome, funded/rejected/ESR status, source document, section type, and citation provenance. This prevents funded and rejected proposal evidence from being confused.

**Acceptance Criteria**:
- [ ] Schema includes required fields for programme, call, topic, year, section type, outcome/source status, source path, and citation anchor.
- [ ] Validation rejects records missing required source-status or programme fields.
- [ ] Example metadata fixtures cover funded, rejected, ESR note, and unknown status.

**Dependencies**: None

**Effort**: 1-2 days (S)

**PRD Reference**: Knowledge Base Ingestion; Source labeling accuracy; Rejected proposals & ESR ingestion

---

### Task 1.3: Implement Docling-based PDF ingestion prototype

**Description**: Build the first ingestion pipeline using Docling to parse proposal PDFs, preserve section structure, and extract text chunks for indexing. Focus on enough fidelity to support section drafting from a small local corpus.

**Acceptance Criteria**:
- [ ] Ingestion accepts at least one proposal PDF and produces structured section-level output.
- [ ] Parsed output preserves document title, headings, section boundaries, and table text where available.
- [ ] Parser failures produce clear errors without corrupting existing indexed data.

**Dependencies**: Task 1.1, Task 1.2

**Effort**: 2-3 days (M)

**PRD Reference**: Knowledge Base Ingestion; Technical Specifications - Parsing

---

### Task 1.4: Build hierarchical chunking and local vector indexing

**Description**: Convert parsed proposal content into hierarchical chunks aligned to proposal sections, then embed and store them locally in Chroma or the selected default vector store.

**Acceptance Criteria**:
- [ ] Chunks retain parent document, section heading, programme, call, and source-status metadata.
- [ ] Local embedding/index creation works without outbound network access.
- [ ] Index can be rebuilt from fixtures and queried in a deterministic test.

**Dependencies**: Task 1.2, Task 1.3

**Effort**: 2-3 days (M)

**PRD Reference**: Knowledge Base Ingestion; Core Architecture; Performance

---

### Task 1.5: Implement source-status-aware retrieval

**Description**: Create retrieval logic that prioritizes funded examples for positive patterns while allowing rejected examples only when relevant or useful as labeled cautionary evidence.

**Acceptance Criteria**:
- [ ] Retriever returns top-k chunks with source-status labels attached to every result.
- [ ] Rejected examples must satisfy the same topical relevance threshold as funded examples unless a lessons-learned flag is enabled.
- [ ] Unit tests cover funded-only, rejected-only, mixed-status, and no-match retrieval scenarios.

**Dependencies**: Task 1.4

**Effort**: 2-3 days (M)

**PRD Reference**: Retrieval Policy; Knowledge Search & Browse; Rejected proposals & ESR ingestion

---

### Task 1.6: Build first section generation workflow

**Description**: Implement a minimal generation workflow that takes a target section type, user intent, call/topic context, retrieved examples, and local LLM output to produce one proposal section draft.

**Acceptance Criteria**:
- [ ] Workflow generates a draft for at least one section type, such as Impact Pathway or Methodology.
- [ ] Generated output includes citations or source references for retrieved evidence.
- [ ] Generation can run using a local model runtime without cloud API calls.

**Dependencies**: Task 1.5

**Effort**: 2-3 days (M)

**PRD Reference**: Agentic Draft Generation; Flow 1; Product Vision

---

### Task 1.7: Render citation and source-status labels in generated output

**Description**: Ensure generated drafts visibly label every cited example as funded, rejected, ESR note, or unknown so users can judge the reliability and context of reused information.

**Acceptance Criteria**:
- [ ] Generated Markdown includes citation markers with source document, section, and source-status label.
- [ ] Automated check fails if any citation lacks a non-empty status tag.
- [ ] Test fixture verifies funded and rejected examples are rendered differently.

**Dependencies**: Task 1.6

**Effort**: 1-2 days (S)

**PRD Reference**: Source labeling accuracy; Agentic Draft Generation; Section Export & Reporting

---

## Sprint 2: MVP Foundation

### Task 2.1: Implement programme drafting profiles

**Description**: Add lightweight v1 programme drafting profiles for terminology, section guidance, expected-output fields, and source-labeling rules. These profiles must be separate from post-v1 scoring rubric profiles.

**Acceptance Criteria**:
- [ ] At least two sample drafting profiles exist, covering Horizon Europe and one additional EU programme.
- [ ] Generation records the active drafting profile used for each draft.
- [ ] Tests prove drafting profiles do not include scoring dimensions or evaluator scoring behavior.

**Dependencies**: Task 1.2, Task 1.6

**Effort**: 1-2 days (S)

**PRD Reference**: Call & Topic Awareness; Programme Drafting Profiles; Drafting profiles vs. scoring rubrics

---

### Task 2.2: Implement call and topic intake

**Description**: Support v1 intake of pasted plaintext and uploaded PDF Work Programme excerpts, extract expected outcomes and topic context, and attach that context to generation prompts.

**Acceptance Criteria**:
- [ ] User can provide call/topic context as pasted text or uploaded PDF excerpt.
- [ ] Extracted context includes programme, topic identifier or title, expected outcomes, and relevant section guidance where available.
- [ ] Generation prompts cite or reference supplied topic requirements in output.

**Dependencies**: Task 1.3, Task 2.1

**Effort**: 2-3 days (M)

**PRD Reference**: Call & Topic Awareness; Flow 1; Success Metrics

---

### Task 2.3: Build proposal ingestion UI and metadata confirmation

**Description**: Add a v1 interface for uploading proposals, reviewing extracted metadata, and confirming source-status fields before content becomes available to generation workflows.

**Acceptance Criteria**:
- [ ] User can upload a proposal document and review extracted metadata before indexing.
- [ ] UI requires programme, call/topic, and source-status confirmation.
- [ ] Confirmed metadata is persisted with the indexed chunks.

**Dependencies**: Task 1.3, Task 1.4

**Effort**: 2-3 days (M)

**PRD Reference**: Flow 3; Knowledge Base Ingestion; User Personas

---

### Task 2.4: Implement incremental indexing and duplicate detection

**Description**: Extend ingestion so new documents can be added without rebuilding the full corpus and duplicates can be detected by title, call ID, source path, or document hash.

**Acceptance Criteria**:
- [ ] Adding a new proposal updates the index without deleting existing proposal chunks.
- [ ] Duplicate title + call ID or document hash triggers a warning and blocks accidental duplicate indexing.
- [ ] Re-indexing a corrected document updates only that document's chunks and metadata.

**Dependencies**: Task 1.4, Task 2.3

**Effort**: 2-3 days (M)

**PRD Reference**: Knowledge Base Ingestion; Flow 3

---

### Task 2.5: Implement offline mode, allowlisting, and network audit logs

**Description**: Enforce the PRD's local-first privacy requirement with egress-denied defaults, explicit allowlisting for non-content outbound requests, and content-safe logging of network attempts.

**Acceptance Criteria**:
- [ ] Offline mode is enabled by default for proposal processing.
- [ ] Any outbound request attempt is logged locally without storing proposal content, prompts, retrieved passages, or generated draft text.
- [ ] Release smoke test fails if a non-opt-in outbound request succeeds.

**Dependencies**: Task 1.1

**Effort**: 2-3 days (M)

**PRD Reference**: Security; Network isolation integrity; Analytics & Monitoring

---

### Task 2.6: Implement content-safe local analytics events

**Description**: Track generation time, iteration count, citation usage, source-status mix, and export events locally without storing proposal content or generated text.

**Acceptance Criteria**:
- [ ] Event schema includes draft start/complete, feedback given, export, generation time, and source-status mix.
- [ ] Tests confirm event payloads do not include raw document content, retrieved passages, or generated draft text.
- [ ] Local analytics are disabled from external export unless a user explicitly exports them.

**Dependencies**: Task 2.5

**Effort**: 1-2 days (S)

**PRD Reference**: Analytics & Monitoring; Security

---

### Task 2.7: Define service boundaries for ingestion, retrieval, generation, and export

**Description**: Refactor the POC into clear service boundaries so UI, ingestion, retrieval, generation, and export can be tested independently before MVP completion.

**Acceptance Criteria**:
- [ ] Ingestion, retrieval, generation, and export interfaces are callable without the UI.
- [ ] Each service has input/output models that include metadata and source-status labels where relevant.
- [ ] Unit tests cover at least one happy path and one error path per service.

**Dependencies**: Task 1.7, Task 2.2, Task 2.4

**Effort**: 2-3 days (M)

**PRD Reference**: Technical Specifications; Feature Requirements; Release Planning MVP

---

## Sprint 3: MVP Completion

### Task 3.1: Build the React section drafting workflow

**Description**: Implement the v1 React/Vite UI workflow where users select a section type, choose a programme drafting profile, provide call/topic context, enter intent or bullets, and generate a section draft using Tailwind CSS and shadcn/ui components.

**Acceptance Criteria**:
- [ ] UI supports section type, drafting profile, call/topic input, and user intent fields.
- [ ] Generate action calls the generation service and displays draft output with citations.
- [ ] Core form controls, buttons, dialogs, tabs, and alerts use shadcn/ui components styled with Tailwind CSS.
- [ ] Empty or incomplete inputs show actionable validation messages.

**Dependencies**: Task 2.2, Task 2.7

**Effort**: 2-3 days (M)

**PRD Reference**: Flow 1; Compatibility; User Personas

---

### Task 3.2: Implement configurable critic loop and user stop control

**Description**: Add the source-grounded critic loop for section drafting, defaulting to 3 revision cycles with user configuration from 1 to 5 and explicit stop at any iteration.

**Acceptance Criteria**:
- [ ] User can set critic iterations between 1 and 5 before generation.
- [ ] User can stop the loop after any completed iteration.
- [ ] Each iteration records what changed and which call/profile requirements were checked.

**Dependencies**: Task 2.1, Task 3.1

**Effort**: 2-3 days (M)

**PRD Reference**: Agentic Draft Generation; Human-in-the-Loop & Iteration; Flow 1

---

### Task 3.3: Implement Markdown and DOCX section export

**Description**: Provide v1 section-level exports that preserve generated text, citations, and source-status labels so users can paste or adapt sections into official proposal templates.

**Acceptance Criteria**:
- [ ] User can export a generated section to Markdown.
- [ ] User can export a generated section to DOCX.
- [ ] Exported files preserve citations and funded/rejected/ESR source-status labels.

**Dependencies**: Task 1.7, Task 3.1

**Effort**: 2-3 days (M)

**PRD Reference**: Section Export & Reporting; Release Planning MVP

---

### Task 3.4: Build citation fidelity and source-label audit harness

**Description**: Create the release audit harness for validating citation fidelity and source labeling against PRD success metrics.

**Acceptance Criteria**:
- [ ] Audit script samples generated sections and lists every citation with source document and status.
- [ ] Automated label check fails if any citation lacks funded, rejected, ESR note, or unknown status.
- [ ] Manual audit template supports checking at least 20 generated sections per release.

**Dependencies**: Task 1.7, Task 3.3

**Effort**: 1-2 days (S)

**PRD Reference**: Success Metrics; Source labeling accuracy; v1 citation fidelity

---

### Task 3.5: Add performance benchmarks for v1 targets

**Description**: Measure indexing, internal retrieval, and section generation latency against the PRD's v1 performance targets on representative local fixtures.

**Acceptance Criteria**:
- [ ] Benchmark measures initial indexing time for a fixture corpus.
- [ ] Benchmark measures retrieval latency for top-k retrieval.
- [ ] Benchmark measures section generation latency and reports model/runtime configuration.

**Dependencies**: Task 2.4, Task 3.1

**Effort**: 1-2 days (S)

**PRD Reference**: Performance; Technical Specifications

---

### Task 3.6: Implement v1 accessibility baseline

**Description**: Add best-effort accessibility for the React/Tailwind/shadcn UI, including keyboard-friendly primary flows, readable contrast, resizable text behavior, and labels for generated reports.

**Acceptance Criteria**:
- [ ] Primary generation and ingestion flows are keyboard navigable with visible focus states.
- [ ] UI labels are present for inputs, buttons, and generated report regions.
- [ ] Tailwind design tokens meet readable contrast targets for primary text, muted text, controls, and status states.
- [ ] Manual accessibility checklist documents remaining gaps for the v1.2 UI refactor.

**Dependencies**: Task 3.1, Task 2.3

**Effort**: 1-2 days (S)

**PRD Reference**: Accessibility; Release Planning v1.2

---

### Task 3.7: Run MVP pilot validation on one active call

**Description**: Validate the MVP with coordinators generating real proposal sections for at least one active call, then capture quality rating, time saved, citation issues, and release-blocking privacy findings.

**Acceptance Criteria**:
- [ ] Pilot includes at least one real call topic and at least three generated section drafts.
- [ ] Users rate each draft and report approximate time saved against manual drafting.
- [ ] Pilot report includes satisfaction, citation issues, performance, network isolation smoke test result, and go/no-go recommendation.

**Dependencies**: Task 3.2, Task 3.3, Task 3.4, Task 3.5

**Effort**: 2-3 days (M)

**PRD Reference**: Success Metrics; MVP Success Criteria; Release Planning MVP

---

## Sprint 4: Post-MVP v1.1

### Task 4.1: Implement scoring rubric profile registry

**Description**: Create versioned scoring rubric profiles for post-v1 evaluator simulation, separated from v1 drafting profiles and tied to programme/call year.

**Acceptance Criteria**:
- [ ] Registry supports versioned rubric profiles such as `horizon-europe-2025.yaml`.
- [ ] Rubric profile includes scoring dimensions, terminology, scale, and comment guidance.
- [ ] Tests prove evaluator workflows require a scoring rubric profile and drafting workflows do not.

**Dependencies**: Task 2.1, Task 3.7

**Effort**: 2-3 days (M)

**PRD Reference**: Scoring Rubric Profiles & Evaluator Simulation; Drafting profiles vs. scoring rubrics

---

### Task 4.2: Build evaluator simulation pipeline

**Description**: Implement post-v1 evaluation for uploaded Part B drafts using the selected programme scoring rubric, comparable funded examples, and clearly tagged rejected examples.

**Acceptance Criteria**:
- [ ] User can upload a draft Part B PDF/DOCX for evaluation.
- [ ] Evaluator selects or requires a scoring rubric profile before scoring.
- [ ] Report includes scores, justifications, red-line suggestions, and source-labeled evidence.

**Dependencies**: Task 4.1, Task 1.5

**Effort**: 2-3 days (M)

**PRD Reference**: Flow 2; Scoring Rubric Profiles & Evaluator Simulation

---

### Task 4.3: Validate evaluator agreement protocol

**Description**: Implement the evaluator agreement measurement process using historical draft/review pairs, two human reviewers, score tolerance, and top-issue overlap.

**Acceptance Criteria**:
- [ ] Evaluation benchmark supports at least 10 historical draft/review pairs per rubric.
- [ ] Agreement calculation counts scores within +/-0.5 and top 3 issue overlap of at least 2 items.
- [ ] Benchmark output reports agreement by rubric and flags dimensions below target.

**Dependencies**: Task 4.2

**Effort**: 2-3 days (M)

**PRD Reference**: Post-v1 evaluator agreement; Success Metrics

---

### Task 4.4: Implement interactive hybrid search and browse

**Description**: Add v1.1 interactive search/browse over indexed proposal sections with BM25 + dense retrieval, filters, previews, and metadata display.

**Acceptance Criteria**:
- [ ] User can search by keyword and semantic query.
- [ ] Filters include programme, call, topic, partner, outcome/source status, and section type.
- [ ] Results show section previews and metadata without exposing hidden document content in logs.

**Dependencies**: Task 2.4, Task 2.7

**Effort**: 2-3 days (M)

**PRD Reference**: Knowledge Search & Browse; Flow 4; Performance

---

### Task 4.5: Add lessons-learned and cautionary examples mode

**Description**: Add an explicit user-controlled mode for surfacing rejected proposals and ESR notes as lessons learned or cautionary examples alongside funded examples.

**Acceptance Criteria**:
- [ ] Search and retrieval UI exposes a clear lessons-learned/cautionary examples control.
- [ ] Rejected proposals and ESR notes remain visibly labeled in every result and generated recommendation.
- [ ] Default mode does not force rejected examples into results unless they meet normal relevance thresholds.

**Dependencies**: Task 4.4, Task 1.5

**Effort**: 1-2 days (S)

**PRD Reference**: Retrieval Policy; Knowledge Search & Browse; Rejected proposals & ESR ingestion

---

### Task 4.6: Implement human-in-the-loop pause, resume, and per-section memory

**Description**: Add controls to pause, resume, and steer agent workflows mid-loop, with feedback persisted into the next iteration for the same section.

**Acceptance Criteria**:
- [ ] User can pause or resume generation/evaluator workflows at defined checkpoints.
- [ ] User feedback is persisted and applied to the next iteration for that section.
- [ ] Per-section memory can be cleared by the user.

**Dependencies**: Task 3.2, Task 4.2

**Effort**: 2-3 days (M)

**PRD Reference**: Human-in-the-Loop & Iteration; Feature Requirements

---

### Task 4.7: Add Funding & Tenders portal HTML intake

**Description**: Extend call/topic awareness to support saved or pasted Funding & Tenders portal HTML topic pages in v1.1 without requiring live portal access during proposal processing.

**Acceptance Criteria**:
- [ ] User can import saved or pasted portal HTML topic content.
- [ ] Parser extracts topic title, expected outcomes, scope, conditions, and destination where present.
- [ ] Import works in offline mode and does not fetch portal content automatically.

**Dependencies**: Task 2.2, Task 2.5

**Effort**: 1-2 days (S)

**PRD Reference**: Call & Topic Awareness; Funding & Tenders Portal glossary

---

## Sprint 5: Future v1.2/v2

### Task 5.1: Implement cross-cutting compliance checks

**Description**: Add automatic checks for ethics, GDPR, gender dimension, open science, security self-assessment, and DoA consistency using rules plus local LLM checks.

**Acceptance Criteria**:
- [ ] Compliance report flags missing or weak cross-cutting sections.
- [ ] Report links each issue to the relevant section or checklist item.
- [ ] Checks run locally and do not send proposal content outside the machine.

**Dependencies**: Task 4.2, Task 4.6

**Effort**: 2-3 days (M)

**PRD Reference**: Cross-Cutting Compliance Checks; Security; Persona 3

---

### Task 5.2: Build wiki-style synthesis workflow

**Description**: Generate and maintain internal Markdown synthesis pages for reusable impact pathways, KPI templates, exploitation strategies, and other institutional patterns.

**Acceptance Criteria**:
- [ ] User can run a synthesis job over selected corpus sections.
- [ ] Output is stored as maintainable Markdown with citations and source-status labels.
- [ ] Synthesis pages are versioned or timestamped so changes can be reviewed.

**Dependencies**: Task 4.4, Task 4.5

**Effort**: 2-3 days (M)

**PRD Reference**: Wiki-Style Synthesis; User Research Findings

---

### Task 5.3: Implement full Part B DOCX/PDF export and page-limit warnings

**Description**: Expand v1 section export into full Part B template-aware DOCX/PDF export with embedded tables, citation audit trail, and page-limit warnings.

**Acceptance Criteria**:
- [ ] User can export a multi-section Part B document to DOCX.
- [ ] PDF export includes citations and source-status labels in a reviewable form.
- [ ] Page-limit warning triggers when generated content exceeds configured section or document limits.

**Dependencies**: Task 3.3, Task 4.6

**Effort**: 2-3 days (M)

**PRD Reference**: Section Export & Reporting; Future Releases v1.2

---

### Task 5.4: Package DGX deployment with Docker

**Description**: Provide Docker-based deployment for consistent Mac-to-DGX operation while preserving offline defaults, local storage paths, and model runtime configuration.

**Acceptance Criteria**:
- [ ] Docker setup starts the application and local services on DGX without external hosting.
- [ ] Container network policy defaults to egress denied during proposal processing.
- [ ] Deployment guide documents model runtime, storage mounts, and offline smoke test.

**Dependencies**: Task 2.5, Task 3.7

**Effort**: 2-3 days (M)

**PRD Reference**: Infrastructure; Security; Compatibility; Release Planning v1.2

---

### Task 5.5: Harden frontend for WCAG 2.1 AA compliance

**Description**: Audit and harden the React/Tailwind/shadcn frontend to meet the PRD's v1.2 WCAG 2.1 AA target across custom flows, generated reports, and export controls.

**Acceptance Criteria**:
- [ ] Accessibility audit identifies React component, Tailwind token, and shadcn/ui usage gaps with chosen remediation approach.
- [ ] Primary flows meet keyboard navigation, contrast, labels, focus states, and resizable text requirements.
- [ ] WCAG 2.1 AA checklist is documented with pass/fail evidence.

**Dependencies**: Task 3.6, Task 4.4

**Effort**: 2-3 days (M)

**PRD Reference**: Accessibility; Release Planning v1.2

---

### Task 5.6: Design multi-user DGX security architecture

**Description**: Prepare v2.0 shared DGX usage with authentication, role-based access control, per-user audit trails, and a dedicated security review before implementation.

**Acceptance Criteria**:
- [ ] Architecture document defines users, roles, permissions, corpus access boundaries, and audit trails.
- [ ] Security review checklist covers confidential partner data, corpus isolation, and content access logging.
- [ ] Implementation tasks are split from design and not merged into v1.x scope.

**Dependencies**: Task 5.4

**Effort**: 2-3 days (M)

**PRD Reference**: Security; Multi-user on DGX; Release Planning v2.0

---

### Task 5.7: Prototype GraphRAG and outcome-feedback learning

**Description**: Prototype graph-based relationships over partners, calls, topics, work packages, outcomes, and ESR feedback to support v2.0 GraphRAG and critic prompt improvement.

**Acceptance Criteria**:
- [ ] Prototype graph model includes partner, topic, proposal, outcome, section, and ESR entities.
- [ ] Query examples demonstrate relationship-aware retrieval beyond flat vector search.
- [ ] Outcome-feedback learning remains advisory and never treats ESR notes as ground truth.

**Dependencies**: Task 4.3, Task 4.4, Task 5.6

**Effort**: 2-3 days (M)

**PRD Reference**: Release Planning v2.0; AI Research Insights; Rejected proposals & ESR ingestion

---

## Dependency Map

### Dependency Table

| Task ID | Task Title | Depends On | Blocks | Wave |
|---------|------------|------------|--------|------|
| 1.1 | Scaffold the local EURPE application workspace | None | 1.3, 2.5 | 1 |
| 1.2 | Define proposal metadata and source-status schema | None | 1.3, 1.4, 2.1 | 1 |
| 1.3 | Implement Docling-based PDF ingestion prototype | 1.1, 1.2 | 1.4, 2.2, 2.3 | 2 |
| 1.4 | Build hierarchical chunking and local vector indexing | 1.2, 1.3 | 1.5, 2.4 | 3 |
| 1.5 | Implement source-status-aware retrieval | 1.4 | 1.6, 4.2, 4.5 | 4 |
| 1.6 | Build first section generation workflow | 1.5 | 1.7, 2.1 | 5 |
| 1.7 | Render citation and source-status labels in generated output | 1.6 | 2.7, 3.3, 3.4 | 6 |
| 2.1 | Implement programme drafting profiles | 1.2, 1.6 | 2.2, 3.2, 4.1 | 6 |
| 2.2 | Implement call and topic intake | 1.3, 2.1 | 2.7, 3.1, 4.7 | 7 |
| 2.3 | Build proposal ingestion UI and metadata confirmation | 1.3, 1.4 | 2.4, 3.6 | 4 |
| 2.4 | Implement incremental indexing and duplicate detection | 1.4, 2.3 | 2.7, 3.5, 4.4 | 5 |
| 2.5 | Implement offline mode, allowlisting, and network audit logs | 1.1 | 2.6, 4.7, 5.4 | 2 |
| 2.6 | Implement content-safe local analytics events | 2.5 | None | 3 |
| 2.7 | Define service boundaries for ingestion, retrieval, generation, and export | 1.7, 2.2, 2.4 | 3.1, 4.4 | 8 |
| 3.1 | Build the React section drafting workflow | 2.2, 2.7 | 3.2, 3.3, 3.5 | 9 |
| 3.2 | Implement configurable critic loop and user stop control | 2.1, 3.1 | 3.7, 4.6 | 10 |
| 3.3 | Implement Markdown and DOCX section export | 1.7, 3.1 | 3.4, 3.7, 5.3 | 10 |
| 3.4 | Build citation fidelity and source-label audit harness | 1.7, 3.3 | 3.7 | 11 |
| 3.5 | Add performance benchmarks for v1 targets | 2.4, 3.1 | 3.7 | 10 |
| 3.6 | Implement v1 accessibility baseline | 3.1, 2.3 | 5.5 | 10 |
| 3.7 | Run MVP pilot validation on one active call | 3.2, 3.3, 3.4, 3.5 | 4.1, 5.4 | 12 |
| 4.1 | Implement scoring rubric profile registry | 2.1, 3.7 | 4.2 | 13 |
| 4.2 | Build evaluator simulation pipeline | 4.1, 1.5 | 4.3, 4.6, 5.1 | 14 |
| 4.3 | Validate evaluator agreement protocol | 4.2 | 5.7 | 15 |
| 4.4 | Implement interactive hybrid search and browse | 2.4, 2.7 | 4.5, 5.2, 5.5, 5.7 | 9 |
| 4.5 | Add lessons-learned and cautionary examples mode | 4.4, 1.5 | 5.2 | 10 |
| 4.6 | Implement human-in-the-loop pause, resume, and per-section memory | 3.2, 4.2 | 5.1, 5.3 | 15 |
| 4.7 | Add Funding & Tenders portal HTML intake | 2.2, 2.5 | None | 8 |
| 5.1 | Implement cross-cutting compliance checks | 4.2, 4.6 | None | 16 |
| 5.2 | Build wiki-style synthesis workflow | 4.4, 4.5 | None | 11 |
| 5.3 | Implement full Part B DOCX/PDF export and page-limit warnings | 3.3, 4.6 | None | 16 |
| 5.4 | Package DGX deployment with Docker | 2.5, 3.7 | 5.6 | 13 |
| 5.5 | Harden frontend for WCAG 2.1 AA compliance | 3.6, 4.4 | None | 11 |
| 5.6 | Design multi-user DGX security architecture | 5.4 | 5.7 | 14 |
| 5.7 | Prototype GraphRAG and outcome-feedback learning | 4.3, 4.4, 5.6 | None | 16 |

### Parallel Execution Groups

**Wave 1 - Start immediately**
- [ ] Task 1.1: Scaffold the local EURPE application workspace
- [ ] Task 1.2: Define proposal metadata and source-status schema

**Wave 2 - Parser and privacy baseline**
- [ ] Task 1.3: Implement Docling-based PDF ingestion prototype
- [ ] Task 2.5: Implement offline mode, allowlisting, and network audit logs

**Wave 3 - Indexing and analytics**
- [ ] Task 1.4: Build hierarchical chunking and local vector indexing
- [ ] Task 2.6: Implement content-safe local analytics events

**Wave 4 - Retrieval and ingestion UI**
- [ ] Task 1.5: Implement source-status-aware retrieval
- [ ] Task 2.3: Build proposal ingestion UI and metadata confirmation

**Wave 5 - First generation and incremental ingestion**
- [ ] Task 1.6: Build first section generation workflow
- [ ] Task 2.4: Implement incremental indexing and duplicate detection

**Wave 6 - Profiles and citation labels**
- [ ] Task 1.7: Render citation and source-status labels in generated output
- [ ] Task 2.1: Implement programme drafting profiles

**Wave 7 - Call/topic intake**
- [ ] Task 2.2: Implement call and topic intake

**Wave 8 - Service boundaries and portal HTML intake**
- [ ] Task 2.7: Define service boundaries for ingestion, retrieval, generation, and export
- [ ] Task 4.7: Add Funding & Tenders portal HTML intake

**Wave 9 - Main UI and search**
- [ ] Task 3.1: Build the React section drafting workflow
- [ ] Task 4.4: Implement interactive hybrid search and browse

**Wave 10 - Critic loop, export, benchmarks, accessibility, lessons mode**
- [ ] Task 3.2: Implement configurable critic loop and user stop control
- [ ] Task 3.3: Implement Markdown and DOCX section export
- [ ] Task 3.5: Add performance benchmarks for v1 targets
- [ ] Task 3.6: Implement v1 accessibility baseline
- [ ] Task 4.5: Add lessons-learned and cautionary examples mode

**Wave 11 - Audit harness, synthesis, accessibility refactor**
- [ ] Task 3.4: Build citation fidelity and source-label audit harness
- [ ] Task 5.2: Build wiki-style synthesis workflow
- [ ] Task 5.5: Harden frontend for WCAG 2.1 AA compliance

**Wave 12 - MVP pilot**
- [ ] Task 3.7: Run MVP pilot validation on one active call

**Wave 13 - Rubric registry and DGX packaging**
- [ ] Task 4.1: Implement scoring rubric profile registry
- [ ] Task 5.4: Package DGX deployment with Docker

**Wave 14 - Evaluator pipeline and multi-user design**
- [ ] Task 4.2: Build evaluator simulation pipeline
- [ ] Task 5.6: Design multi-user DGX security architecture

**Wave 15 - Evaluator validation and human-in-loop**
- [ ] Task 4.3: Validate evaluator agreement protocol
- [ ] Task 4.6: Implement human-in-the-loop pause, resume, and per-section memory

**Wave 16 - Future feature completion**
- [ ] Task 5.1: Implement cross-cutting compliance checks
- [ ] Task 5.3: Implement full Part B DOCX/PDF export and page-limit warnings
- [ ] Task 5.7: Prototype GraphRAG and outcome-feedback learning

### Critical Path

`Task 1.2 -> Task 1.3 -> Task 1.4 -> Task 1.5 -> Task 1.6 -> Task 1.7 -> Task 2.7 -> Task 3.1 -> Task 3.3 -> Task 3.4 -> Task 3.7 -> Task 4.1 -> Task 4.2 -> Task 4.6 -> Task 5.3`

**Critical Path Estimate**: 26-39 work days, assuming one engineer and sequential execution of critical-path work.

**Bottlenecks**:
- Docling extraction quality affects indexing, retrieval, citation fidelity, and export.
- Source-status schema quality affects almost every downstream feature.
- The generation service boundary is a gate for UI, export, and later evaluator work.
- MVP pilot validation blocks confident v1.1 rubric/evaluator investment.

## Flagged Ambiguous Requirements

| Requirement | What Needs Clarification |
|-------------|--------------------------|
| Model runtime | PRD lists 14B-70B local LLMs, but the exact default model, quantization, context length, and Mac/DGX model split are not specified. |
| Corpus composition | PRD assumes ~40 proposals, but the funded/rejected/ESR distribution and number of scanned/low-quality PDFs are not specified. |
| Drafting profiles | Initial list of EU programmes beyond Horizon Europe, Horizon 2020, Digital Europe, and CEF needs prioritization. |
| Export templates | v1 section DOCX export is defined, but the exact DOCX template, citation style, and section formatting rules need sample files. |
| Offline enforcement | PRD states egress-denied mode, but implementation needs a chosen macOS firewall/process allowlisting approach. |
| User satisfaction measurement | PRD sets >=4/5 satisfaction, but survey wording and scoring cadence need confirmation. |
| Active call pilot | MVP validation requires one active call, but the target call and section types are not yet named. |

## Requirement Coverage

| PRD Requirement | Covering Tasks |
|-----------------|----------------|
| Knowledge Base Ingestion | Task 1.2, Task 1.3, Task 1.4, Task 2.3, Task 2.4 |
| Call & Topic Awareness | Task 2.1, Task 2.2, Task 4.7 |
| Agentic Draft Generation | Task 1.5, Task 1.6, Task 1.7, Task 3.1, Task 3.2 |
| Scoring Rubric Profiles & Evaluator Simulation | Task 4.1, Task 4.2, Task 4.3 |
| Knowledge Search & Browse | Task 4.4, Task 4.5 |
| Human-in-the-Loop & Iteration | Task 3.2, Task 4.6 |
| Cross-Cutting Compliance Checks | Task 5.1 |
| Wiki-Style Synthesis | Task 5.2 |
| Section Export & Reporting | Task 3.3, Task 5.3 |
| Security and network isolation | Task 2.5, Task 2.6, Task 5.4, Task 5.6 |
| Accessibility | Task 3.6, Task 5.5 |
| Performance | Task 3.5 |

## Technical Notes

- v1 should avoid live Funding & Tenders portal scraping. Use pasted text, uploaded PDF excerpts, or saved HTML to preserve offline guarantees.
- Treat programme drafting profiles and scoring rubric profiles as separate config families. Drafting profiles guide generation; scoring rubric profiles drive post-v1 evaluator simulation.
- Store analytics and audit logs as metadata only. Do not log proposal content, prompts, retrieved passages, generated text, or partner-confidential data.
- Build the MVP around a single complete section-drafting workflow before broadening into evaluator simulation or full Part B export.
