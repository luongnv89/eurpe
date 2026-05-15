import { useState } from "react";

import { ShieldCheck } from "lucide-react";

import { Button } from "@/components/ui/button";
import { IngestWizard } from "@/features/ingest/IngestWizard";

type View = "home" | "ingest";

/**
 * Top-level app shell. A single ``useState`` view discriminator stands in
 * for routing for now — once Task 3.1 introduces more than two views we
 * will pull in react-router-dom. Keeping it inline keeps the dep tree
 * small for the Wave-4 prototype.
 */
export default function App() {
  const [view, setView] = useState<View>("home");

  if (view === "ingest") {
    return (
      <main className="min-h-screen bg-background">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-6 py-4">
          <Button variant="ghost" onClick={() => setView("home")}>
            ← Home
          </Button>
          <span className="inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs text-muted-foreground">
            <ShieldCheck className="h-3.5 w-3.5" />
            Local-only · Offline by default
          </span>
        </div>
        <IngestWizard />
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
        <div className="flex justify-center gap-3">
          <Button onClick={() => setView("ingest")}>Ingest a proposal</Button>
          <Button variant="outline" disabled>
            Browse corpus
          </Button>
        </div>
      </div>
    </main>
  );
}
