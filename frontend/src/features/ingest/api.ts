/**
 * Typed fetch helpers for the ``/api/ingestion`` routes.
 *
 * URLs are relative on purpose — Vite's dev server proxies ``/api`` to
 * ``http://127.0.0.1:8765`` (see vite.config.ts). The same relative paths
 * also work in production where the React build is served alongside the
 * FastAPI app.
 *
 * Errors are normalised into a single ``IngestionError`` shape so the
 * components don't have to special-case ``response.detail`` vs
 * ``response.detail[0].msg`` (the FastAPI 422 layout).
 */

export interface ParseResponse {
  parse_token: string;
  source_path: string;
  parsed_at: string;
  suggested: SuggestedDraft;
  page_count: number | null;
  title: string | null;
  section_count: number;
}

export interface SuggestedDraft {
  programme?: string;
  call_id?: string;
  topic_id?: string;
  proposal_title?: string;
}

export interface ConfirmRequest {
  parse_token: string;
  programme: string;
  call_id: string;
  topic_id?: string | null;
  year: number;
  outcome: string;
  proposal_title?: string | null;
  consortium_acronym?: string | null;
  language?: string;
}

export interface ConfirmResponse {
  chunks_added: number;
  collection: string;
  sidecar_path: string;
}

export interface EnumsResponse {
  programme: string[];
  source_status: string[];
}

export class IngestionError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(detail);
    this.name = "IngestionError";
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
      return loc ? `${loc}: ${first?.msg ?? "validation error"}` : (first?.msg ?? "validation error");
    }
    return JSON.stringify(body);
  } catch {
    return response.statusText || `request failed with status ${response.status}`;
  }
}

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new IngestionError(response.status, await readErrorDetail(response));
  }
  return (await response.json()) as T;
}

export async function fetchEnums(signal?: AbortSignal): Promise<EnumsResponse> {
  const response = await fetch("/api/ingestion/enums", { signal });
  return parseJsonOrThrow<EnumsResponse>(response);
}

export async function uploadForParse(file: File, signal?: AbortSignal): Promise<ParseResponse> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("/api/ingestion/parse", {
    method: "POST",
    body: form,
    signal,
  });
  return parseJsonOrThrow<ParseResponse>(response);
}

export async function confirmIngestion(
  body: ConfirmRequest,
  signal?: AbortSignal,
): Promise<ConfirmResponse> {
  const response = await fetch("/api/ingestion/confirm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  return parseJsonOrThrow<ConfirmResponse>(response);
}
