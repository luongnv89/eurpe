import { ShieldCheck } from "lucide-react";

import { Button } from "@/components/ui/button";

export default function App() {
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
          <Button>Start a draft</Button>
          <Button variant="outline">Browse corpus</Button>
        </div>
      </div>
    </main>
  );
}
