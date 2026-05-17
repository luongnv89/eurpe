import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

interface WorkspaceHeaderProps {
  eyebrow: string;
  title: string;
  description: string;
  actions?: ReactNode;
  className?: string;
}

/**
 * Editorial-style page header used by every workspace page. Provides:
 *   - amber rule + uppercase eyebrow as wayfinding context
 *   - large serif title
 *   - one-line description (kept short by design — billboard rule)
 *   - optional right-aligned actions slot
 */
export function WorkspaceHeader({
  eyebrow,
  title,
  description,
  actions,
  className,
}: WorkspaceHeaderProps) {
  return (
    <header
      className={cn(
        "border-b border-brand-hairline bg-background px-6 pt-10 pb-8 lg:px-12",
        className,
      )}
    >
      <div className="mx-auto max-w-6xl flex flex-wrap items-end justify-between gap-6">
        <div className="space-y-3">
          <p className="inline-flex items-center gap-2 text-xs font-medium uppercase tracking-[0.22em] text-brand-navy/60">
            <span aria-hidden="true" className="h-px w-6 bg-brand-amber" />
            {eyebrow}
          </p>
          <h1 className="font-display text-3xl tracking-tight text-brand-navy sm:text-4xl">
            {title}
          </h1>
          <p className="max-w-2xl text-sm leading-relaxed text-brand-navy/70">
            {description}
          </p>
        </div>
        {actions && (
          <div className="flex flex-wrap items-center gap-2">{actions}</div>
        )}
      </div>
    </header>
  );
}

interface Step {
  label: string;
  state: "done" | "active" | "todo";
}

/**
 * Numbered step rail rendered above a multi-step workflow. Each step
 * shows a state-tinted amber dot plus a label. The rail is decorative —
 * the workflow logic owns advancement; this component reflects state.
 */
export function StepRail({ steps }: { steps: Step[] }) {
  return (
    <ol
      aria-label="Workflow steps"
      className="mx-auto flex max-w-6xl flex-wrap items-center gap-3 px-6 pt-8 lg:px-12"
    >
      {steps.map((step, i) => {
        const n = (i + 1).toString().padStart(2, "0");
        return (
          <li key={step.label} className="flex items-center gap-3">
            <span
              className={cn(
                "step-dot transition-colors",
                step.state === "todo" &&
                  "bg-brand-parchment text-brand-navy/55 ring-1 ring-brand-navy/10",
                step.state === "done" && "bg-brand-amber/25 text-brand-navy",
              )}
              aria-current={step.state === "active" ? "step" : undefined}
            >
              {n}
            </span>
            <span
              className={cn(
                "text-sm",
                step.state === "active"
                  ? "font-medium text-brand-navy"
                  : "text-brand-navy/60",
              )}
            >
              {step.label}
            </span>
            {i < steps.length - 1 && (
              <span
                aria-hidden="true"
                className="ml-1 h-px w-8 bg-brand-hairline"
              />
            )}
          </li>
        );
      })}
    </ol>
  );
}
