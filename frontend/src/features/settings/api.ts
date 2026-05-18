/**
 * Typed fetch helpers for the ``/api/runtime`` routes (issue #79).
 *
 * Mirrors the pattern in ``frontend/src/features/drafting/api.ts``:
 * relative URLs (Vite proxies ``/api`` to the FastAPI app), one
 * ``RuntimeError`` per failure mode, and a shared error-detail
 * extractor.
 */

export interface RuntimeStatus {
  /** Runtime key (e.g. 'ollama', 'vllm', 'mlx'). */
  runtime: string;
  /** Human-readable runtime name. */
  display_name: string;
  /** URL that was probed. */
  endpoint: string;
  /** True when the runtime responded to the health probe. */
  available: boolean;
  /** Model identifiers reported by the runtime. Empty when unavailable. */
  models: string[];
  /** Human-readable error when the runtime is unavailable. */
  error: string | null;
}

export interface AllRuntimesResponse {
  /** Status for each supported runtime. */
  runtimes: RuntimeStatus[];
  /** The runtime key currently configured in config.yaml. */
  active_runtime: string;
}

export interface InstallInstructions {
  /** Human-readable runtime name. */
  title: string;
  /** Newline-separated installation steps. */
  steps: string;
  /** Link to official documentation. */
  docs_url: string;
}

export interface RuntimeInstructionsResponse {
  instructions: InstallInstructions;
}

export class RuntimeError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(detail);
    this.name = "RuntimeError";
    this.status = status;
    this.detail = detail;
  }
}

async function readErrorDetail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
    return JSON.stringify(body);
  } catch {
    return response.statusText || `request failed with status ${response.status}`;
  }
}

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new RuntimeError(response.status, await readErrorDetail(response));
  }
  return (await response.json()) as T;
}

export async function fetchRuntimeStatus(signal?: AbortSignal): Promise<RuntimeStatus> {
  const response = await fetch("/api/runtime/status", { signal });
  return parseJsonOrThrow<RuntimeStatus>(response);
}

export async function fetchAllRuntimes(signal?: AbortSignal): Promise<AllRuntimesResponse> {
  const response = await fetch("/api/runtime/all", { signal });
  return parseJsonOrThrow<AllRuntimesResponse>(response);
}

export async function fetchRuntimeInstructions(
  runtime: string,
  signal?: AbortSignal,
): Promise<RuntimeInstructionsResponse> {
  const response = await fetch(`/api/runtime/instructions/${runtime}`, { signal });
  return parseJsonOrThrow<RuntimeInstructionsResponse>(response);
}

// ---------------------------------------------------------------------------
// Cloud provider connection test (issue #80)
// ---------------------------------------------------------------------------

export interface CloudProviderTestRequest {
  provider: string;
  model: string;
  api_key: string;
}

export interface CloudProviderTestResponse {
  success: boolean;
  message: string;
  model_confirmed: string | null;
  error_detail: string | null;
}

export async function testCloudProviderConnection(
  body: CloudProviderTestRequest,
  signal?: AbortSignal,
): Promise<CloudProviderTestResponse> {
  const response = await fetch("/api/cloud/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  return parseJsonOrThrow<CloudProviderTestResponse>(response);
}

// ---------------------------------------------------------------------------
// Local model and embedding test (issue #81)
// ---------------------------------------------------------------------------

export interface LocalModelTestRequest {
  runtime: string;
  model: string;
  base_url?: string | null;
}

export interface LocalEmbeddingTestRequest {
  runtime: string;
  model: string;
  base_url?: string | null;
}

export interface LocalModelTestResponse {
  success: boolean;
  message: string;
  error_detail: string | null;
}

export async function testLocalModel(
  body: LocalModelTestRequest,
  signal?: AbortSignal,
): Promise<LocalModelTestResponse> {
  const response = await fetch("/api/runtime/test-model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  return parseJsonOrThrow<LocalModelTestResponse>(response);
}

export async function testLocalEmbedding(
  body: LocalEmbeddingTestRequest,
  signal?: AbortSignal,
): Promise<LocalModelTestResponse> {
  const response = await fetch("/api/runtime/test-embedding", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  return parseJsonOrThrow<LocalModelTestResponse>(response);
}
