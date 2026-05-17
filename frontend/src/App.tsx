import { useState } from "react";

import { ShieldCheck } from "lucide-react";

import { Button } from "@/components/ui/button";
import { DraftingWorkspace } from "@/features/drafting/DraftingWorkspace";
import { IngestWizard } from "@/features/ingest/IngestWizard";

type View = "home" | "ingest" | "draft";

/**
 * Top-level app shell. A single ``useState`` view discriminator stands in
 * for routing — Task 3.1 introduces the second non-home view (drafting)
 * so we now manage three states. Once a fourth view lands (e.g., the
 * corpus browser stub on the home page) we will pull in react-router-dom
 * rather than continue extending this enum.
 */
export default function App() {
  const [view, setView] = useState<View>("home");

  if (view === "ingest") {
    return (
      <main className="min-h-screen bg-background">
        <TopBar onHome={() => setView("home")} />
        <IngestWizard />
      </main>
    );
  }

  if (view === "draft") {
    return (
      <main className="min-h-screen bg-background">
        <TopBar onHome={() => setView("home")} />
        <DraftingWorkspace />
      </main>
    );
  }

  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-8">
      <div className="max-w-xl text-center space-y-6">
        <div className="inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs text-muted-foreground">
          <ShieldCheck className="h-3.5 w-3.5" />
          <span>Local-only · Offline by default</span>
        </div>
        <h1 className="text-4xl font-semibold tracking-tight">EURPE</h1>
        <p className="text-muted-foreground">
          EU Research Proposal Expert — a fully-local AI assistant for drafting Horizon Europe and
          related programme proposals from your own past submissions.
        </p>
        <div className="flex flex-wrap justify-center gap-3">
          <Button onClick={() => setView("ingest")}>Ingest a proposal</Button>
          <Button onClick={() => setView("draft")}>Draft a section</Button>
          <Button variant="outline" disabled>
            Browse corpus
          </Button>
        </div>
      </div>
    </main>
  );
}

/**
 * Shared top-bar so both feature views show the same "← Home" affordance
 * and offline-mode reminder without duplicating markup.
 */
function TopBar({ onHome }: { onHome: () => void }) {
  return (
    <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
      <Button variant="ghost" onClick={onHome}>
        ← Home
      </Button>
      <span className="inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs text-muted-foreground">
        <ShieldCheck className="h-3.5 w-3.5" />
        Local-only · Offline by default
      </span>
    </div>
  );
}
