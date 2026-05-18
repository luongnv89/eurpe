/**
 * Typed fetch helpers for the Settings page.
 *
 * Combines configuration editing (`/api/config`), local runtime probes
 * (`/api/runtime`), and explicit cloud-provider connection tests
 * (`/api/cloud/test`). All URLs are relative so Vite proxies them to the
 * local FastAPI app.
 */

// ---------------------------------------------------------------------------
// Configuration editing (issue #74)
// ---------------------------------------------------------------------------

export interface NetworkAllowlistEntry {
  host: string;
  port: number;
  reason: string;
}

export interface ModelsConfig {
  runtime: string;
  llm_model: string;
  embedding_model: string;
  ollama_base_url: string;
  llm_base_url: string | null;
  llm_api_key_env: string | null;
}

export interface ConfigResponse {
  corpus_path: string;
  index_path: string;
  runtime_dir: string;
  offline_mode: boolean;
  log_level: string;
  models: ModelsConfig;
  network_allowlist: NetworkAllowlistEntry[];
}

export interface ConfigUpdateRequest {
  corpus_path?: string | null;
  index_path?: string | null;
  runtime_dir?: string | null;
  offline_mode?: boolean | null;
  log_level?: string | null;
  models?: Partial<ModelsConfig> | null;
  network_allowlist?: NetworkAllowlistEntry[] | null;
}

export interface ConfigUpdateResponse {
  ok: boolean;
  config: ConfigResponse;
}

// ---------------------------------------------------------------------------
// Runtime health and model listing (issues #79 / #81)
// ---------------------------------------------------------------------------

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

export class SettingsError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "SettingsError";
    this.status = status;
    this.detail = detail;
  }
}

// Backwards-compatible name for the runtime helper introduced by issue #79.
export class RuntimeError extends SettingsError {
  constructor(status: number, detail: string) {
    super(status, detail);
    this.name = "RuntimeError";
  }
}

async function readErrorDetail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail) && body.detail.length > 0) {
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
    throw new SettingsError(response.status, await readErrorDetail(response));
  }
  return (await response.json()) as T;
}

export async function fetchConfig(signal?: AbortSignal): Promise<ConfigResponse> {
  const response = await fetch("/api/config", { signal });
  return parseJsonOrThrow<ConfigResponse>(response);
}

export async function updateConfig(
  body: ConfigUpdateRequest,
  signal?: AbortSignal,
): Promise<ConfigUpdateResponse> {
  const response = await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  return parseJsonOrThrow<ConfigUpdateResponse>(response);
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
