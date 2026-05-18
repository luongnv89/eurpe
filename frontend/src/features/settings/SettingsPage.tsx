import { useEffect, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Cpu,
  Database,
  FolderTree,
  RefreshCw,
  Sparkles,
} from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { WorkspaceHeader } from "@/components/WorkspaceHeader";
import {
  fetchAllRuntimes,
  fetchRuntimeInstructions,
  type AllRuntimesResponse,
  type InstallInstructions,
  type RuntimeStatus,
} from "./api";

interface SettingItem {
  Icon: typeof Cpu;
  label: string;
  value: string;
  hint: string;
}

const LLM_SETTINGS: SettingItem[] = [
  {
    Icon: Cpu,
    label: "LLM runtime",
    value: "Ollama · llama3.1:8b",
    hint: "Edit config.yaml → models.llm_model to switch.",
  },
  {
    Icon: Sparkles,
    label: "Critic loop",
    value: "Max 5 iterations",
    hint: "Server-enforced ceiling; per-draft override in the workspace.",
  },
];

const EMBEDDING_SETTINGS: SettingItem[] = [
  {
    Icon: Database,
    label: "Embedding model",
    value: "nomic-embed-text",
    hint: "Edit config.yaml → models.embedding_model to switch. Used for vector ingestion and semantic search.",
  },
];

const INDEX_SETTINGS: SettingItem[] = [
  {
    Icon: FolderTree,
    label: "Index path",
    value: "./data/index",
    hint: "Delete the folder to reset the corpus.",
  },
];

// TODO(#83): Re-enable Network & Security card when outbound features
// (EU Funding & Tenders auto-fill, cloud LLM fallback) are ready.
// {
//   Icon: ShieldCheck,
//   label: "Network",
//   value: "Local-only",
//   hint: "EU Funding & Tenders auto-fill is the only optional outbound call.",
// },

/**
 * Settings page — read-only summary of effective config in v1. The
 * server-side config lives in config.yaml and is currently edited there;
 * a writeable Settings UI is on the roadmap.
 *
 * Issue #79: detects local runtime availability and shows installed models
 * plus installation instructions when a runtime is unreachable.
 */
export function SettingsPage() {
  const [allRuntimes, setAllRuntimes] = useState<AllRuntimesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [instructions, setInstructions] = useState<Record<string, InstallInstructions>>({});
  const [expandedRuntime, setExpandedRuntime] = useState<string | null>(null);

  const loadRuntimes = async () => {
    setLoading(true);
    try {
      const data = await fetchAllRuntimes();
      setAllRuntimes(data);

      // Pre-fetch instructions for unavailable runtimes
      const instMap: Record<string, InstallInstructions> = {};
      await Promise.all(
        data.runtimes
          .filter((r) => !r.available)
          .map(async (r) => {
            try {
              const resp = await fetchRuntimeInstructions(r.runtime);
              instMap[r.runtime] = resp.instructions;
            } catch {
              // Instructions are best-effort
            }
          }),
      );
      setInstructions(instMap);
    } catch {
      // Silently fail — the static cards still render
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadRuntimes();
  }, []);

  return (
    <section aria-labelledby="settings-heading">
      <WorkspaceHeader
        eyebrow="Workspace · Settings"
        title="Effective configuration"
        description="A read-only view of how EURPE is currently configured on this machine. Edit config.yaml to change these values."
      />

      <div className="mx-auto max-w-6xl space-y-8 px-6 py-12 lg:px-12">
        <h2 id="settings-heading" className="sr-only">
          Effective configuration
        </h2>

        {/* Local Runtime Detection (issue #79) */}
        <div>
          <div className="mb-3 flex items-center justify-between">
            <h3 className="font-display text-sm font-medium uppercase tracking-[0.12em] text-brand-navy/55">
              Local runtime
            </h3>
            <button
              type="button"
              onClick={loadRuntimes}
              disabled={loading}
              className="inline-flex items-center gap-1 text-xs text-brand-navy/55 hover:text-brand-navy disabled:opacity-40"
              aria-label="Refresh runtime status"
            >
              <RefreshCw
                className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`}
                aria-hidden="true"
              />
              Refresh
            </button>
          </div>

          {loading && !allRuntimes ? (
            <Card className="border-brand-navy/10 shadow-editorial">
              <CardContent className="py-8">
                <p className="text-center text-sm text-brand-navy/55">
                  Detecting local runtimes…
                </p>
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
                  onToggle={() =>
                    setExpandedRuntime((prev) =>
                      prev === rt.runtime ? null : rt.runtime,
                    )
                  }
                />
              ))}
            </div>
          ) : null}
        </div>

        <div>
          <h3 className="mb-3 font-display text-sm font-medium uppercase tracking-[0.12em] text-brand-navy/55">
            LLM generation
          </h3>
          <ul className="grid gap-4 sm:grid-cols-2">
            {LLM_SETTINGS.map(({ Icon, label, value, hint }) => (
              <li key={label}>
                <Card className="border-brand-navy/10 shadow-editorial">
                  <CardHeader className="pb-2">
                    <CardTitle className="flex items-center gap-2 text-sm font-medium text-brand-navy/65">
                      <span
                        aria-hidden="true"
                        className="inline-flex h-7 w-7 items-center justify-center rounded bg-brand-parchment text-brand-navy ring-1 ring-brand-navy/10"
                      >
                        <Icon className="h-3.5 w-3.5" aria-hidden="true" />
                      </span>
                      {label}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="font-display text-xl text-brand-navy">
                      {value}
                    </p>
                    <p className="mt-2 text-xs leading-relaxed text-brand-navy/60">
                      {hint}
                    </p>
                  </CardContent>
                </Card>
              </li>
            ))}
          </ul>
        </div>

        <div>
          <h3 className="mb-3 font-display text-sm font-medium uppercase tracking-[0.12em] text-brand-navy/55">
            Embeddings &amp; vector search
          </h3>
          <ul className="grid gap-4 sm:grid-cols-2">
            {EMBEDDING_SETTINGS.map(({ Icon, label, value, hint }) => (
              <li key={label}>
                <Card className="border-brand-navy/10 shadow-editorial">
                  <CardHeader className="pb-2">
                    <CardTitle className="flex items-center gap-2 text-sm font-medium text-brand-navy/65">
                      <span
                        aria-hidden="true"
                        className="inline-flex h-7 w-7 items-center justify-center rounded bg-brand-parchment text-brand-navy ring-1 ring-brand-navy/10"
                      >
                        <Icon className="h-3.5 w-3.5" aria-hidden="true" />
                      </span>
                      {label}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="font-display text-xl text-brand-navy">
                      {value}
                    </p>
                    <p className="mt-2 text-xs leading-relaxed text-brand-navy/60">
                      {hint}
                    </p>
                  </CardContent>
                </Card>
              </li>
            ))}
          </ul>
        </div>

        <div>
          <h3 className="mb-3 font-display text-sm font-medium uppercase tracking-[0.12em] text-brand-navy/55">
            Index
          </h3>
          <ul className="grid gap-4 sm:grid-cols-2">
            {INDEX_SETTINGS.map(({ Icon, label, value, hint }) => (
              <li key={label}>
                <Card className="border-brand-navy/10 shadow-editorial">
                  <CardHeader className="pb-2">
                    <CardTitle className="flex items-center gap-2 text-sm font-medium text-brand-navy/65">
                      <span
                        aria-hidden="true"
                        className="inline-flex h-7 w-7 items-center justify-center rounded bg-brand-parchment text-brand-navy ring-1 ring-brand-navy/10"
                      >
                        <Icon className="h-3.5 w-3.5" aria-hidden="true" />
                      </span>
                      {label}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="font-display text-xl text-brand-navy">
                      {value}
                    </p>
                    <p className="mt-2 text-xs leading-relaxed text-brand-navy/60">
                      {hint}
                    </p>
                  </CardContent>
                </Card>
              </li>
            ))}
          </ul>
        </div>

        <Card className="border-brand-amber/40 bg-brand-amber/[0.08] shadow-editorial">
          <CardHeader>
            <CardTitle className="font-display text-brand-navy">
              About this build
            </CardTitle>
            <CardDescription className="text-brand-navy/70">
              EURPE is local-first. None of the values above are shipped to a
              remote service — they are summarised here only to help you
              understand the state of your own machine.
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

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[11px] uppercase tracking-[0.18em] text-brand-navy/55">
        {label}
      </dt>
      <dd className="mt-1 font-mono text-brand-navy">{value}</dd>
    </div>
  );
}

// ---------------------------------------------------------------------------
// RuntimeCard — per-runtime status, model list, and install instructions
// ---------------------------------------------------------------------------

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
  return (
    <Card
      className={[
        "border-brand-navy/10 shadow-editorial transition-colors",
        isActive ? "ring-1 ring-brand-amber/50" : "",
      ].join(" ")}
    >
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
          <div className="flex items-center gap-2">
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
        </div>
        <CardDescription className="font-mono text-xs text-brand-navy/50">
          {status.endpoint}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {status.available ? (
          <div>
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
              <p className="text-sm text-brand-navy/60">
                Runtime is running but no models are installed.
              </p>
            )}
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
