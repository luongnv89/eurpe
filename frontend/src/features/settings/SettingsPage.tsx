import { Cpu, FolderTree, ShieldCheck, Sparkles } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { WorkspaceHeader } from "@/components/WorkspaceHeader";

interface SettingItem {
  Icon: typeof Cpu;
  label: string;
  value: string;
  hint: string;
}

const SETTINGS: SettingItem[] = [
  {
    Icon: Cpu,
    label: "Local LLM",
    value: "Ollama · llama3.1:8b",
    hint: "Edit config.yaml → llm.model to switch.",
  },
  {
    Icon: FolderTree,
    label: "Index path",
    value: "./data/index",
    hint: "Delete the folder to reset the corpus.",
  },
  {
    Icon: Sparkles,
    label: "Critic loop",
    value: "Max 5 iterations",
    hint: "Server-enforced ceiling; per-draft override in the workspace.",
  },
  {
    Icon: ShieldCheck,
    label: "Network",
    value: "Local-only",
    hint: "EU Funding & Tenders auto-fill is the only optional outbound call.",
  },
];

/**
 * Settings page — read-only summary of effective config in v1. The
 * server-side config lives in config.yaml and is currently edited there;
 * a writeable Settings UI is on the roadmap.
 */
export function SettingsPage() {
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

        <ul className="grid gap-4 sm:grid-cols-2">
          {SETTINGS.map(({ Icon, label, value, hint }) => (
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

        <Card className="border-brand-amber/40 bg-brand-amber/8 shadow-editorial">
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
