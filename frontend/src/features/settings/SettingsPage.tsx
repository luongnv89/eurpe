import { useCallback, useEffect, useState } from "react";
import {
  AlertCircle,
  Check,
  CheckCircle2,
  Cloud,
  Cpu,
  Eye,
  EyeOff,
  FolderOpen,
  FolderTree,
  Globe,
  Loader2,
  RefreshCw,
  Save,
  ShieldCheck,
} from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { WorkspaceHeader } from "@/components/WorkspaceHeader";
import {
  fetchAllRuntimes,
  fetchConfig,
  fetchRuntimeInstructions,
  testCloudProviderConnection,
  testLocalEmbedding,
  testLocalModel,
  type AllRuntimesResponse,
  type CloudProviderTestResponse,
  type ConfigResponse,
  type InstallInstructions,
  type LocalModelTestResponse,
  type NetworkAllowlistEntry,
  type RuntimeStatus,
  updateConfig,
} from "./api";

const RUNTIMES = [
  { value: "ollama", label: "Ollama (local)" },
  { value: "lmstudio", label: "LM Studio (local)" },
  { value: "llamacpp", label: "llama.cpp (local)" },
  { value: "vllm", label: "vLLM (self-hosted)" },
  { value: "openai", label: "OpenAI (cloud)" },
  { value: "openrouter", label: "OpenRouter (cloud)" },
  { value: "groq", label: "Groq (cloud)" },
  { value: "anthropic", label: "Anthropic (cloud)" },
  { value: "gemini", label: "Gemini (cloud)" },
];

const LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"];

const DEFAULT_LLM_MODELS: Record<string, string> = {
  ollama: "llama3.1:8b",
  lmstudio: "local-model",
  llamacpp: "local-model",
  vllm: "meta-llama/Llama-3.1-8B-Instruct",
  openai: "gpt-4o-mini",
  openrouter: "openai/gpt-4o-mini",
  groq: "llama-3.1-8b-instant",
  anthropic: "claude-3-5-sonnet-20241022",
  gemini: "gemini-1.5-flash",
};

const CLOUD_PROVIDERS = [
  { key: "openai", label: "OpenAI", defaultModel: "gpt-4o" },
  { key: "anthropic", label: "Anthropic", defaultModel: "claude-sonnet-4-20250514" },
  { key: "gemini", label: "Gemini", defaultModel: "gemini-2.5-flash" },
  { key: "openrouter", label: "OpenRouter", defaultModel: "openai/gpt-4o" },
  { key: "groq", label: "Groq", defaultModel: "llama-3.3-70b-versatile" },
];

interface SettingsForm {
  corpus_path: string;
  index_path: string;
  runtime_dir: string;
  offline_mode: boolean;
  log_level: string;
  models: {
    runtime: string;
    llm_model: string;
    embedding_model: string;
    ollama_base_url: string;
    llm_base_url: string;
    llm_api_key_env: string;
  };
  network_allowlist: NetworkAllowlistEntry[];
}

function formFromConfig(cfg: ConfigResponse): SettingsForm {
  return {
    corpus_path: cfg.corpus_path,
    index_path: cfg.index_path,
    runtime_dir: cfg.runtime_dir,
    offline_mode: cfg.offline_mode,
    log_level: cfg.log_level,
    models: {
      runtime: cfg.models.runtime,
      llm_model: cfg.models.llm_model,
      embedding_model: cfg.models.embedding_model,
      ollama_base_url: cfg.models.ollama_base_url,
      llm_base_url: cfg.models.llm_base_url ?? "",
      llm_api_key_env: cfg.models.llm_api_key_env ?? "",
    },
    network_allowlist: cfg.network_allowlist.map((e) => ({ ...e })),
  };
}

export function SettingsPage() {
  const [form, setForm] = useState<SettingsForm | null>(null);
  const [loadingConfig, setLoadingConfig] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [allRuntimes, setAllRuntimes] = useState<AllRuntimesResponse | null>(null);
  const [loadingRuntimes, setLoadingRuntimes] = useState(true);
  const [instructions, setInstructions] = useState<Record<string, InstallInstructions>>({});
  const [expandedRuntime, setExpandedRuntime] = useState<string | null>(null);

  const loadConfig = useCallback(async () => {
    setLoadingConfig(true);
    setError(null);
    try {
      const cfg = await fetchConfig();
      setForm(formFromConfig(cfg));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load configuration");
    } finally {
      setLoadingConfig(false);
    }
  }, []);

  const loadRuntimes = useCallback(async () => {
    setLoadingRuntimes(true);
    try {
      const data = await fetchAllRuntimes();
      setAllRuntimes(data);

      const instMap: Record<string, InstallInstructions> = {};
      await Promise.all(
        data.runtimes
          .filter((r) => !r.available)
          .map(async (r) => {
            try {
              const resp = await fetchRuntimeInstructions(r.runtime);
              instMap[r.runtime] = resp.instructions;
            } catch {
              // Instructions are best-effort.
            }
          }),
      );
      setInstructions(instMap);
    } catch {
      // Runtime diagnostics are supplementary; config editing still works.
    } finally {
      setLoadingRuntimes(false);
    }
  }, []);

  useEffect(() => {
    loadConfig();
    loadRuntimes();
  }, [loadConfig, loadRuntimes]);

  const updateField = <K extends keyof SettingsForm>(key: K, value: SettingsForm[K]) => {
    if (!form) return;
    setForm({ ...form, [key]: value });
  };

  const updateModelField = <K extends keyof SettingsForm["models"]>(
    key: K,
    value: SettingsForm["models"][K],
  ) => {
    if (!form) return;
    const nextModels = { ...form.models, [key]: value };

    if (key === "runtime") {
      nextModels.llm_model = DEFAULT_LLM_MODELS[value] ?? nextModels.llm_model;
    }

    setForm({ ...form, models: nextModels });
  };

  const handleSave = async () => {
    if (!form) return;
    setSaving(true);
    setError(null);
    setSaved(false);
    try {
      const resp = await updateConfig({
        corpus_path: form.corpus_path,
        index_path: form.index_path,
        runtime_dir: form.runtime_dir,
        offline_mode: form.offline_mode,
        log_level: form.log_level,
        models: {
          runtime: form.models.runtime,
          llm_model: form.models.llm_model,
          embedding_model: form.models.embedding_model,
          ollama_base_url: form.models.ollama_base_url,
          llm_base_url: form.models.llm_base_url || null,
          llm_api_key_env: form.models.llm_api_key_env || null,
        },
        network_allowlist: form.network_allowlist,
      });
      setForm(formFromConfig(resp.config));
      setSaved(true);
      await loadRuntimes();
      setTimeout(() => setSaved(false), 3000);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to save configuration");
    } finally {
      setSaving(false);
    }
  };

  const addAllowlistEntry = () => {
    if (!form) return;
    setForm({
      ...form,
      network_allowlist: [...form.network_allowlist, { host: "", port: 443, reason: "" }],
    });
  };

  const updateAllowlistEntry = (index: number, field: string, value: string | number) => {
    if (!form) return;
    const next = [...form.network_allowlist];
    next[index] = { ...next[index], [field]: value };
    setForm({ ...form, network_allowlist: next });
  };

  const removeAllowlistEntry = (index: number) => {
    if (!form) return;
    setForm({
      ...form,
      network_allowlist: form.network_allowlist.filter((_, i) => i !== index),
    });
  };

  if (loadingConfig && !form) {
    return (
      <section aria-labelledby="settings-heading" className="flex items-center justify-center py-24">
        <Loader2 className="h-8 w-8 animate-spin text-brand-amber" />
      </section>
    );
  }

  if (!form) {
    return (
      <section aria-labelledby="settings-heading" className="py-12">
        <Alert variant="destructive" className="mx-auto max-w-2xl">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Failed to load configuration</AlertTitle>
          <AlertDescription>{error ?? "Unknown error"}</AlertDescription>
        </Alert>
      </section>
    );
  }

  return (
    <section aria-labelledby="settings-heading">
      <WorkspaceHeader
        eyebrow="Workspace · Settings"
        title="Configuration"
        description="Edit YAML-backed settings, inspect local runtimes, and test provider connectivity from this machine."
      />

      <div className="mx-auto max-w-6xl space-y-8 px-6 py-12 lg:px-12">
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {saved && (
          <Alert className="border-green-200 bg-green-50">
            <Check className="h-4 w-4 text-green-600" />
            <AlertTitle className="text-green-800">Configuration saved</AlertTitle>
            <AlertDescription className="text-green-700">
              Changes have been written to config.yaml.
            </AlertDescription>
          </Alert>
        )}

        <Card className="border-brand-navy/10 shadow-editorial">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 font-display text-brand-navy">
              <Cpu className="h-5 w-5" />
              LLM Provider &amp; Model
            </CardTitle>
            <CardDescription>
              Select the generation backend and model. Cloud providers require an API key set as an
              environment variable and a network_allowlist entry.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="runtime">Runtime</Label>
              <Select value={form.models.runtime} onValueChange={(v) => updateModelField("runtime", v)}>
                <SelectTrigger id="runtime">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {RUNTIMES.map((r) => (
                    <SelectItem key={r.value} value={r.value}>
                      {r.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="llm_model">LLM Model</Label>
              <Input
                id="llm_model"
                value={form.models.llm_model}
                onChange={(e) => updateModelField("llm_model", e.target.value)}
                placeholder="e.g. llama3.1:8b"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="embedding_model">Embedding Model</Label>
              <Input
                id="embedding_model"
                value={form.models.embedding_model}
                onChange={(e) => updateModelField("embedding_model", e.target.value)}
                placeholder="e.g. nomic-embed-text"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="ollama_base_url">Ollama Base URL</Label>
              <Input
                id="ollama_base_url"
                value={form.models.ollama_base_url}
                onChange={(e) => updateModelField("ollama_base_url", e.target.value)}
                placeholder="http://localhost:11434"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="llm_base_url">LLM Base URL Override</Label>
              <Input
                id="llm_base_url"
                value={form.models.llm_base_url}
                onChange={(e) => updateModelField("llm_base_url", e.target.value)}
                placeholder="Leave blank for provider default"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="llm_api_key_env">API Key Env Var Name</Label>
              <Input
                id="llm_api_key_env"
                value={form.models.llm_api_key_env}
                onChange={(e) => updateModelField("llm_api_key_env", e.target.value)}
                placeholder="e.g. OPENAI_API_KEY"
              />
              <p className="text-xs text-brand-navy/50">
                Set the actual secret in your shell environment, never in config.yaml.
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="border-brand-navy/10 shadow-editorial">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 font-display text-brand-navy">
              <FolderTree className="h-5 w-5" />
              Paths
            </CardTitle>
            <CardDescription>Directories for the corpus, vector index, and runtime state.</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="corpus_path">Corpus Path</Label>
              <Input
                id="corpus_path"
                value={form.corpus_path}
                onChange={(e) => updateField("corpus_path", e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="index_path">Index Path</Label>
              <Input
                id="index_path"
                value={form.index_path}
                onChange={(e) => updateField("index_path", e.target.value)}
              />
            </div>
            <div className="space-y-2 sm:col-span-2">
              <Label htmlFor="runtime_dir">Runtime Dir</Label>
              <Input
                id="runtime_dir"
                value={form.runtime_dir}
                onChange={(e) => updateField("runtime_dir", e.target.value)}
              />
            </div>
          </CardContent>
        </Card>

        <Card className="border-brand-navy/10 shadow-editorial">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 font-display text-brand-navy">
              <ShieldCheck className="h-5 w-5" />
              Network &amp; Security
            </CardTitle>
            <CardDescription>
              Offline mode blocks all outbound traffic by default. Add allowlist entries for cloud
              providers.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex items-center gap-4">
              <Label htmlFor="offline_mode" className="flex-1">
                Offline Mode
              </Label>
              <button
                id="offline_mode"
                role="switch"
                aria-checked={form.offline_mode}
                onClick={() => updateField("offline_mode", !form.offline_mode)}
                className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors ${
                  form.offline_mode ? "bg-brand-amber" : "bg-gray-200"
                }`}
              >
                <span
                  className={`pointer-events-none inline-block h-5 w-5 rounded-full bg-white shadow ring-0 transition-transform ${
                    form.offline_mode ? "translate-x-5" : "translate-x-0"
                  }`}
                />
              </button>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Network Allowlist</Label>
                <Button variant="outline" size="sm" onClick={addAllowlistEntry}>
                  <Globe className="mr-1 h-3 w-3" />
                  Add Entry
                </Button>
              </div>
              {form.network_allowlist.length === 0 && (
                <p className="text-sm text-brand-navy/50">
                  No outbound connections allowed. Only loopback (localhost) traffic is permitted.
                </p>
              )}
              {form.network_allowlist.map((entry, i) => (
                <div key={i} className="grid gap-2 rounded-lg border border-brand-navy/10 p-3 sm:grid-cols-12">
                  <div className="sm:col-span-3">
                    <Label className="text-xs">Host</Label>
                    <Input
                      value={entry.host}
                      onChange={(e) => updateAllowlistEntry(i, "host", e.target.value)}
                      placeholder="api.openai.com"
                    />
                  </div>
                  <div className="sm:col-span-2">
                    <Label className="text-xs">Port</Label>
                    <Input
                      type="number"
                      value={entry.port}
                      onChange={(e) => updateAllowlistEntry(i, "port", parseInt(e.target.value) || 443)}
                    />
                  </div>
                  <div className="sm:col-span-6">
                    <Label className="text-xs">Reason</Label>
                    <Input
                      value={entry.reason}
                      onChange={(e) => updateAllowlistEntry(i, "reason", e.target.value)}
                      placeholder="Operator-approved OpenAI generation"
                    />
                  </div>
                  <div className="flex items-end sm:col-span-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeAllowlistEntry(i)}
                      className="text-red-500 hover:text-red-700"
                    >
                      ×
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="border-brand-navy/10 shadow-editorial">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 font-display text-brand-navy">
              <FolderOpen className="h-5 w-5" />
              General
            </CardTitle>
            <CardDescription>Logging and miscellaneous settings.</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="log_level">Log Level</Label>
              <Select value={form.log_level} onValueChange={(v) => updateField("log_level", v)}>
                <SelectTrigger id="log_level">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {LOG_LEVELS.map((level) => (
                    <SelectItem key={level} value={level}>
                      {level}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        <div className="flex items-center justify-end gap-3">
          <Button variant="outline" onClick={loadConfig}>
            <RefreshCw className="mr-1 h-4 w-4" />
            Reload
          </Button>
          <Button onClick={handleSave} disabled={saving}>
            {saving ? (
              <>
                <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                <Save className="mr-1 h-4 w-4" />
                Save Configuration
              </>
            )}
          </Button>
        </div>

        <RuntimeDiagnostics
          allRuntimes={allRuntimes}
          loading={loadingRuntimes}
          instructions={instructions}
          expandedRuntime={expandedRuntime}
          onRefresh={loadRuntimes}
          onToggleRuntime={(runtime) =>
            setExpandedRuntime((prev) => (prev === runtime ? null : runtime))
          }
        />

        <CloudProviderTestCard />

        <Card className="border-brand-amber/40 bg-brand-amber/[0.08] shadow-editorial">
          <CardHeader>
            <CardTitle className="font-display text-brand-navy">About this build</CardTitle>
            <CardDescription className="text-brand-navy/70">
              EURPE is local-first. Configuration is stored in config.yaml on this machine and is
              never sent to a remote service unless you explicitly enable and allowlist a provider.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <dl className="grid gap-3 text-sm sm:grid-cols-3">
              <Stat label="Version" value="v0.1.0" />
              <Stat label="Frontend" value="Vite + React" />
              <Stat label="Backend" value="FastAPI + Chroma" />
            </dl>
          </CardContent>
        </Card>
      </div>
    </section>
  );
}

function RuntimeDiagnostics({
  allRuntimes,
  loading,
  instructions,
  expandedRuntime,
  onRefresh,
  onToggleRuntime,
}: {
  allRuntimes: AllRuntimesResponse | null;
  loading: boolean;
  instructions: Record<string, InstallInstructions>;
  expandedRuntime: string | null;
  onRefresh: () => void;
  onToggleRuntime: (runtime: string) => void;
}) {
  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <h3 className="font-display text-sm font-medium uppercase tracking-[0.12em] text-brand-navy/55">
          Local runtime
        </h3>
        <button
          type="button"
          onClick={onRefresh}
          disabled={loading}
          className="inline-flex items-center gap-1 text-xs text-brand-navy/55 hover:text-brand-navy disabled:opacity-40"
          aria-label="Refresh runtime status"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} aria-hidden="true" />
          Refresh
        </button>
      </div>

      {loading && !allRuntimes ? (
        <Card className="border-brand-navy/10 shadow-editorial">
          <CardContent className="py-8">
            <p className="text-center text-sm text-brand-navy/55">Detecting local runtimes…</p>
          </CardContent>
        </Card>
      ) : allRuntimes ? (
        <div className="space-y-4">
          {allRuntimes.runtimes.map((rt) => (
            <RuntimeCard
              key={rt.runtime}
              status={rt}
              isActive={rt.runtime === allRuntimes.active_runtime}
              instructions={instructions[rt.runtime] ?? null}
              isExpanded={expandedRuntime === rt.runtime}
              onToggle={() => onToggleRuntime(rt.runtime)}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[11px] uppercase tracking-[0.18em] text-brand-navy/55">{label}</dt>
      <dd className="mt-1 font-mono text-brand-navy">{value}</dd>
    </div>
  );
}

function RuntimeCard({
  status,
  isActive,
  instructions,
  isExpanded,
  onToggle,
}: {
  status: RuntimeStatus;
  isActive: boolean;
  instructions: InstallInstructions | null;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const [testingModel, setTestingModel] = useState(false);
  const [testingEmbedding, setTestingEmbedding] = useState(false);
  const [modelResult, setModelResult] = useState<LocalModelTestResponse | null>(null);
  const [embeddingResult, setEmbeddingResult] = useState<LocalModelTestResponse | null>(null);
  const [selectedModel, setSelectedModel] = useState(status.models[0] ?? "");

  const handleTestModel = async (model: string) => {
    if (!model) return;
    setTestingModel(true);
    setModelResult(null);
    try {
      const resp = await testLocalModel({ runtime: status.runtime, model, base_url: status.endpoint });
      setModelResult(resp);
    } catch (err) {
      setModelResult({
        success: false,
        message: "Request failed",
        error_detail: err instanceof Error ? err.message : "Unknown error",
      });
    } finally {
      setTestingModel(false);
    }
  };

  const handleTestEmbedding = async (model: string) => {
    if (!model) return;
    setTestingEmbedding(true);
    setEmbeddingResult(null);
    try {
      const resp = await testLocalEmbedding({ runtime: status.runtime, model, base_url: status.endpoint });
      setEmbeddingResult(resp);
    } catch (err) {
      setEmbeddingResult({
        success: false,
        message: "Request failed",
        error_detail: err instanceof Error ? err.message : "Unknown error",
      });
    } finally {
      setTestingEmbedding(false);
    }
  };

  return (
    <Card className={["border-brand-navy/10 shadow-editorial transition-colors", isActive ? "ring-1 ring-brand-amber/50" : ""].join(" ")}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-sm font-medium text-brand-navy/65">
            <span
              aria-hidden="true"
              className="inline-flex h-7 w-7 items-center justify-center rounded bg-brand-parchment text-brand-navy ring-1 ring-brand-navy/10"
            >
              <Cpu className="h-3.5 w-3.5" aria-hidden="true" />
            </span>
            {status.display_name}
            {isActive && (
              <span className="rounded bg-brand-amber/15 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-brand-amber-600">
                Active
              </span>
            )}
          </CardTitle>
          {status.available ? (
            <span className="inline-flex items-center gap-1 text-xs font-medium text-green-700">
              <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
              Running
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 text-xs font-medium text-red-600">
              <AlertCircle className="h-3.5 w-3.5" aria-hidden="true" />
              Not available
            </span>
          )}
        </div>
        <CardDescription className="font-mono text-xs text-brand-navy/50">{status.endpoint}</CardDescription>
      </CardHeader>
      <CardContent>
        {status.available ? (
          <div className="space-y-4">
            {status.models.length > 0 ? (
              <div>
                <p className="mb-2 text-xs font-medium text-brand-navy/65">
                  Installed models ({status.models.length})
                </p>
                <ul className="flex flex-wrap gap-1.5">
                  {status.models.map((m) => (
                    <li key={m}>
                      <span className="inline-flex rounded bg-brand-parchment px-2 py-0.5 font-mono text-xs text-brand-navy ring-1 ring-brand-navy/10">
                        {m}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <p className="text-sm text-brand-navy/60">Runtime is running but no models are installed.</p>
            )}

            <div className="flex flex-wrap gap-2 pt-2">
              <div className="min-w-[200px] flex-1">
                <label className="mb-1 block text-xs font-medium text-brand-navy/65">Model to test</label>
                <div className="flex gap-2">
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="flex-1 rounded-md border border-brand-navy/15 bg-white px-3 py-1.5 text-xs text-brand-navy focus:border-brand-amber focus:outline-none focus:ring-1 focus:ring-brand-amber"
                  >
                    {status.models.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    onClick={() => handleTestModel(selectedModel)}
                    disabled={testingModel || !selectedModel}
                    className="inline-flex items-center gap-1 rounded-md bg-brand-amber px-3 py-1.5 text-xs font-medium text-brand-navy shadow-amber transition-colors hover:bg-brand-amber-600 hover:text-white disabled:pointer-events-none disabled:opacity-50"
                  >
                    {testingModel ? (
                      <>
                        <Loader2 className="h-3 w-3 animate-spin" />
                        Testing…
                      </>
                    ) : (
                      "Test Model"
                    )}
                  </button>
                </div>
                {modelResult && <TestResultDisplay result={modelResult} />}
              </div>

              <div className="min-w-[200px] flex-1">
                <label className="mb-1 block text-xs font-medium text-brand-navy/65">Embedding model</label>
                <div className="flex gap-2">
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="flex-1 rounded-md border border-brand-navy/15 bg-white px-3 py-1.5 text-xs text-brand-navy focus:border-brand-amber focus:outline-none focus:ring-1 focus:ring-brand-amber"
                  >
                    {status.models.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    onClick={() => handleTestEmbedding(selectedModel)}
                    disabled={testingEmbedding || !selectedModel}
                    className="inline-flex items-center gap-1 rounded-md bg-brand-amber px-3 py-1.5 text-xs font-medium text-brand-navy shadow-amber transition-colors hover:bg-brand-amber-600 hover:text-white disabled:pointer-events-none disabled:opacity-50"
                  >
                    {testingEmbedding ? (
                      <>
                        <Loader2 className="h-3 w-3 animate-spin" />
                        Testing…
                      </>
                    ) : (
                      "Test Embedding"
                    )}
                  </button>
                </div>
                {embeddingResult && <TestResultDisplay result={embeddingResult} />}
              </div>
            </div>
          </div>
        ) : (
          <div>
            <p className="mb-3 text-sm text-brand-navy/70">
              {status.error || "This runtime is not reachable on your machine."}
            </p>
            {instructions && (
              <button
                type="button"
                onClick={onToggle}
                className="text-xs font-medium text-brand-amber-600 hover:text-brand-amber hover:underline"
              >
                {isExpanded ? "Hide" : "Show"} installation instructions →
              </button>
            )}
            {isExpanded && instructions && (
              <div className="mt-4 rounded-md border border-brand-navy/10 bg-brand-parchment/60 p-4">
                <h4 className="mb-2 font-display text-sm font-medium text-brand-navy">
                  How to set up {instructions.title}
                </h4>
                <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-brand-navy/80">
                  {instructions.steps}
                </pre>
                <a
                  href={instructions.docs_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-3 inline-block text-xs font-medium text-brand-amber-600 hover:text-brand-amber hover:underline"
                >
                  Official documentation →
                </a>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function CloudProviderTestCard() {
  const [provider, setProvider] = useState(CLOUD_PROVIDERS[0].key);
  const [model, setModel] = useState(CLOUD_PROVIDERS[0].defaultModel);
  const [apiKey, setApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState<CloudProviderTestResponse | null>(null);

  const selectedProvider = CLOUD_PROVIDERS.find((p) => p.key === provider)!;

  const handleProviderChange = (newProvider: string) => {
    setProvider(newProvider);
    const p = CLOUD_PROVIDERS.find((cp) => cp.key === newProvider)!;
    setModel(p.defaultModel);
    setResult(null);
  };

  const handleTest = async () => {
    if (!apiKey.trim()) return;
    setTesting(true);
    setResult(null);
    try {
      const resp = await testCloudProviderConnection({ provider, model, api_key: apiKey });
      setResult(resp);
    } catch (err) {
      setResult({
        success: false,
        message: "Request failed",
        model_confirmed: null,
        error_detail: err instanceof Error ? err.message : "Unknown error",
      });
    } finally {
      setTesting(false);
    }
  };

  const canTest = apiKey.trim().length > 0 && !testing;

  return (
    <div>
      <h3 className="mb-3 font-display text-sm font-medium uppercase tracking-[0.12em] text-brand-navy/55">
        Cloud provider
      </h3>
      <Card className="border-brand-navy/10 shadow-editorial">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-sm font-medium text-brand-navy/65">
            <span
              aria-hidden="true"
              className="inline-flex h-7 w-7 items-center justify-center rounded bg-brand-parchment text-brand-navy ring-1 ring-brand-navy/10"
            >
              <Cloud className="h-3.5 w-3.5" aria-hidden="true" />
            </span>
            Test connection
          </CardTitle>
          <CardDescription className="text-xs text-brand-navy/50">
            Verify your API key works with the selected provider and model. Uses a minimal request
            (1 token) to avoid unnecessary charges.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <label className="mb-1 block text-xs font-medium text-brand-navy/65">Provider</label>
              <select
                value={provider}
                onChange={(e) => handleProviderChange(e.target.value)}
                className="w-full rounded-md border border-brand-navy/15 bg-white px-3 py-2 text-sm text-brand-navy focus:border-brand-amber focus:outline-none focus:ring-1 focus:ring-brand-amber"
              >
                {CLOUD_PROVIDERS.map((p) => (
                  <option key={p.key} value={p.key}>
                    {p.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="mb-1 block text-xs font-medium text-brand-navy/65">Model</label>
              <input
                type="text"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                placeholder="e.g. gpt-4o"
                className="w-full rounded-md border border-brand-navy/15 bg-white px-3 py-2 text-sm text-brand-navy placeholder:text-brand-navy/30 focus:border-brand-amber focus:outline-none focus:ring-1 focus:ring-brand-amber"
              />
            </div>

            <div>
              <label className="mb-1 block text-xs font-medium text-brand-navy/65">API key</label>
              <div className="relative">
                <input
                  type={showKey ? "text" : "password"}
                  value={apiKey}
                  onChange={(e) => {
                    setApiKey(e.target.value);
                    setResult(null);
                  }}
                  placeholder={`Enter your ${selectedProvider.label} API key`}
                  className="w-full rounded-md border border-brand-navy/15 bg-white px-3 py-2 pr-10 text-sm text-brand-navy placeholder:text-brand-navy/30 focus:border-brand-amber focus:outline-none focus:ring-1 focus:ring-brand-amber"
                />
                <button
                  type="button"
                  onClick={() => setShowKey((s) => !s)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-brand-navy/40 hover:text-brand-navy"
                  aria-label={showKey ? "Hide API key" : "Show API key"}
                >
                  {showKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            <button
              type="button"
              onClick={handleTest}
              disabled={!canTest}
              className="inline-flex items-center justify-center gap-2 rounded-md bg-brand-amber px-4 py-2 text-sm font-medium text-brand-navy shadow-amber transition-colors hover:bg-brand-amber-600 hover:text-white disabled:pointer-events-none disabled:opacity-50"
            >
              {testing ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Testing…
                </>
              ) : (
                "Test Connection"
              )}
            </button>

            {result && <CloudTestResult result={result} />}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function CloudTestResult({ result }: { result: CloudProviderTestResponse }) {
  return (
    <div
      className={[
        "flex items-start gap-2 rounded-md border p-3 text-sm",
        result.success ? "border-green-200 bg-green-50 text-green-800" : "border-red-200 bg-red-50 text-red-800",
      ].join(" ")}
      role="status"
      aria-live="polite"
    >
      {result.success ? (
        <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" />
      ) : (
        <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
      )}
      <div>
        <p className="font-medium">{result.message}</p>
        {result.model_confirmed && <p className="mt-1 text-xs opacity-80">Model: {result.model_confirmed}</p>}
        {result.error_detail && <p className="mt-1 font-mono text-xs opacity-80">{result.error_detail}</p>}
      </div>
    </div>
  );
}

function TestResultDisplay({ result }: { result: LocalModelTestResponse }) {
  return (
    <div
      className={[
        "mt-2 flex items-start gap-1.5 rounded-md border p-2 text-xs",
        result.success ? "border-green-200 bg-green-50 text-green-800" : "border-red-200 bg-red-50 text-red-800",
      ].join(" ")}
      role="status"
      aria-live="polite"
    >
      {result.success ? (
        <CheckCircle2 className="mt-0.5 h-3 w-3 shrink-0" />
      ) : (
        <AlertCircle className="mt-0.5 h-3 w-3 shrink-0" />
      )}
      <div>
        <p className="font-medium">{result.message}</p>
        {result.error_detail && <p className="mt-0.5 font-mono text-[10px] opacity-80">{result.error_detail}</p>}
      </div>
    </div>
  );
}
