/**
 * Typed fetch helpers for the ``/api/generation`` routes.
 *
 * Mirrors ``frontend/src/features/ingest/api.ts`` so a maintainer
 * picking up either feature recognises the shape: relative URLs (Vite
 * proxies ``/api`` to the FastAPI app on 127.0.0.1:8765), one
 * ``GenerationError`` per failure mode, and a shared error-detail
 * extractor that handles the FastAPI 422 envelope.
 *
 * The wire types deliberately mirror :class:`GenerateSectionResponse`
 * (and friends) in ``src/eurpe/api/schemas.py`` field-for-field. When a
 * field is added on the server it must be added here too; both files
 * carry a single source of truth comment so the linker is obvious.
 */

export interface GenerationEnumsResponse {
  /** SectionType enum values (e.g., 'methodology', 'impact'). */
  section_type: string[];
  /** Programme enum values (e.g., 'horizon_europe'). */
  programme: string[];
}

export interface DraftingProfileSummary {
  programme: string;
  name: string;
}

export interface ProfilesResponse {
  profiles: DraftingProfileSummary[];
}

export interface GenerateSectionRequest {
  section_type: string;
  user_intent: string;
  call_context?: string;
  target_programme?: string | null;
  profile_programme?: string | null;
  top_k_examples?: number;
  lessons_learned?: boolean;
}

export interface CitationPayload {
  citation_id: number;
  source_status: string;
  programme: string;
  call_id: string;
  proposal_title: string | null;
  section_heading: string | null;
  page: number | null;
  chunk_id: string;
  snippet: string;
}

/**
 * One critic-loop iteration record (Task 3.2 / issue #16).
 *
 * Mirrors :class:`eurpe.api.schemas.IterationRecordPayload`. The
 * implicit first pass is NOT included in the iterations list — the
 * draft body itself is iteration 1. Each entry in ``iterations``
 * records a critic+regenerate pass with the changes summary and the
 * deterministically-computed list of call/profile requirements the
 * critic was instructed to check (AC #3).
 */
export interface IterationRecordPayload {
  iteration_index: number;
  changes_summary: string;
  requirements_checked: string[];
  critique_text: string;
  generated_at: string;
}

export interface GenerateSectionResponse {
  section_type: string;
  text: string;
  citations: CitationPayload[];
  model: string;
  generated_at: string;
  drafting_profile: string | null;
  /**
   * Cumulative critic-loop history. Empty for single-pass drafts;
   * grows by one entry per /section/iterate call. Surfaced so the UI
   * can render an iteration timeline alongside the draft.
   */
  iterations: IterationRecordPayload[];
}

export class GenerationError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(detail);
    this.name = "GenerationError";
    this.status = status;
    this.detail = detail;
  }
}

async function readErrorDetail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail) && body.detail.length > 0) {
      // FastAPI 422 returns a list of {loc, msg, type} objects. Surface
      // the first one's location + message so the operator can fix the
      // specific field that failed.
      const first = body.detail[0];
      const loc = Array.isArray(first?.loc)
        ? first.loc.filter((p: unknown) => typeof p === "string").join(".")
        : "";
      return loc
        ? `${loc}: ${first?.msg ?? "validation error"}`
        : (first?.msg ?? "validation error");
    }
    return JSON.stringify(body);
  } catch {
    return response.statusText || `request failed with status ${response.status}`;
  }
}

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new GenerationError(response.status, await readErrorDetail(response));
  }
  return (await response.json()) as T;
}

export async function fetchGenerationEnums(
  signal?: AbortSignal,
): Promise<GenerationEnumsResponse> {
  const response = await fetch("/api/generation/enums", { signal });
  return parseJsonOrThrow<GenerationEnumsResponse>(response);
}

export async function fetchProfiles(signal?: AbortSignal): Promise<ProfilesResponse> {
  const response = await fetch("/api/generation/profiles", { signal });
  return parseJsonOrThrow<ProfilesResponse>(response);
}

export async function generateSection(
  body: GenerateSectionRequest,
  signal?: AbortSignal,
): Promise<GenerateSectionResponse> {
  const response = await fetch("/api/generation/section", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  return parseJsonOrThrow<GenerateSectionResponse>(response);
}

/**
 * Request body for POST /api/generation/section/iterate.
 *
 * Mirrors :class:`eurpe.api.schemas.IterateSectionRequest`. The
 * workspace owns the loop: it stores the original request inputs
 * (section_type, intent, programme, profile, top_k, lessons_learned),
 * the iteration cap (1-5), and the most recent draft, and posts this
 * body once per iteration. Stop = stop calling (AC #2).
 */
export interface IterateSectionRequest {
  section_type: string;
  user_intent: string;
  call_context?: string;
  target_programme?: string | null;
  profile_programme?: string | null;
  top_k_examples?: number;
  lessons_learned?: boolean;
  max_iterations: number;
  prior_draft: GenerateSectionResponse;
}

export interface IterateSectionResponse {
  draft: GenerateSectionResponse;
  iteration_index: number;
  max_iterations: number;
  /**
   * True when this iteration is the last permitted (iteration_index
   * == max_iterations). Advisory — the UI uses this to disable the
   * "Refine" button. The user can also stop earlier by clicking
   * "Accept draft" instead of "Refine" at any time (AC #2).
   */
  stopped: boolean;
}

export async function iterateSection(
  body: IterateSectionRequest,
  signal?: AbortSignal,
): Promise<IterateSectionResponse> {
  const response = await fetch("/api/generation/section/iterate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  return parseJsonOrThrow<IterateSectionResponse>(response);
}
