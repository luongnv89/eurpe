import { Link } from "react-router-dom";
import { FileStack, Upload } from "lucide-react";

import { Button } from "@/components/ui/button";
import { WorkspaceHeader } from "@/components/WorkspaceHeader";

/**
 * Corpus browser — v1 placeholder. The /api/corpus listing endpoint is
 * not yet wired (Task 4.x), so this page renders the affordances and an
 * empty state with a clear "what to do next" path: ingest a proposal.
 *
 * Replacing this with the real listing is a single component swap; the
 * shell, header, and routing are already correct.
 */
export function CorpusBrowser() {
  return (
    <section aria-labelledby="corpus-heading">
      <WorkspaceHeader
        eyebrow="Workspace · Corpus"
        title="Indexed proposals"
        description="Read-only browser for what is in your local Chroma index — proposals, sections, and chunks."
        actions={
          <Button asChild variant="amber">
            <Link to="/ingest">
              <Upload className="h-4 w-4" aria-hidden="true" />
              Ingest a proposal
            </Link>
          </Button>
        }
      />

      <div className="mx-auto max-w-6xl px-6 py-12 lg:px-12">
        <EmptyState />
      </div>
    </section>
  );
}

function EmptyState() {
  return (
    <div className="rounded-xl border border-dashed border-brand-navy/15 bg-brand-parchment/40 p-12 text-center">
      <span
        aria-hidden="true"
        className="mx-auto inline-flex h-12 w-12 items-center justify-center rounded-full bg-white ring-1 ring-brand-navy/10"
      >
        <FileStack className="h-5 w-5 text-brand-navy" aria-hidden="true" />
      </span>
      <h2 className="font-display mt-5 text-xl text-brand-navy">
        Browser coming in v1.1
      </h2>
      <p className="mx-auto mt-3 max-w-md text-sm leading-relaxed text-brand-navy/70">
        The listing API for indexed proposals is on the v1.1 roadmap. For now,
        the drafting workspace pulls retrieval examples directly from your
        local Chroma store — you don't need this page to draft.
      </p>
      <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
        <Button asChild variant="amber">
          <Link to="/ingest">
            <Upload className="h-4 w-4" aria-hidden="true" />
            Add another proposal
          </Link>
        </Button>
        <Button asChild variant="outline">
          <Link to="/draft">Start drafting</Link>
        </Button>
      </div>
    </div>
  );
}
