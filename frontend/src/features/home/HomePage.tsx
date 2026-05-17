import { Link } from "react-router-dom";
import {
  ArrowUpRight,
  FileStack,
  PenLine,
  ShieldCheck,
  Sparkles,
  Upload,
} from "lucide-react";

import { Button } from "@/components/ui/button";

/**
 * Editorial landing: oversized serif headline + amber rule, three entry
 * tiles (Ingest, Draft, Corpus), then a privacy assurance ribbon. The
 * "Browse corpus" tile is shown but flagged "v2" since the v1 backend
 * doesn't expose the listing endpoint yet — keeps users oriented without
 * a disabled-button dead end.
 */
export function HomePage() {
  return (
    <div className="relative">
      <Hero />
      <EntryTiles />
      <PrivacyRibbon />
      <Footer />
    </div>
  );
}

function Hero() {
  return (
    <section
      aria-labelledby="home-heading"
      className="relative px-6 pt-16 pb-24 lg:pt-24 lg:pb-32"
    >
      <div className="mx-auto max-w-5xl">
        <p className="inline-flex items-center gap-2 text-xs font-medium uppercase tracking-[0.22em] text-brand-navy/60">
          <span className="h-px w-8 bg-brand-amber" aria-hidden="true" />
          A local AI assistant for Horizon Europe
        </p>

        <h1
          id="home-heading"
          className="font-display mt-6 text-5xl leading-[1.05] tracking-tight text-brand-navy sm:text-6xl lg:text-7xl"
        >
          Draft proposals from{" "}
          <span className="amber-rule">your own past work</span>—
          <br className="hidden sm:block" />
          <span className="text-brand-navy/70">never the cloud.</span>
        </h1>

        <p className="mt-8 max-w-2xl text-lg leading-relaxed text-brand-navy/75">
          EURPE turns the proposals you have already written into a private,
          searchable evidence base. Index past submissions, then have a local
          LLM stitch a first pass for any section — grounded in your own
          language, never leaving your machine.
        </p>

        <div className="mt-10 flex flex-wrap items-center gap-3">
          <Button asChild variant="amber" size="lg">
            <Link to="/draft">
              <Sparkles className="h-4 w-4" aria-hidden="true" />
              Draft a section
            </Link>
          </Button>
          <Button asChild variant="outline" size="lg">
            <Link to="/ingest">
              <Upload className="h-4 w-4" aria-hidden="true" />
              Ingest a proposal
            </Link>
          </Button>
          <span className="ml-2 inline-flex items-center gap-2 rounded-full border border-brand-navy/10 bg-white px-3 py-1.5 text-xs text-brand-navy/70">
            <ShieldCheck
              className="h-3.5 w-3.5 text-brand-amber-600"
              aria-hidden="true"
            />
            Runs entirely on this machine
          </span>
        </div>
      </div>

      {/* Editorial side rail with kerned metric numerals */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute right-6 top-32 hidden xl:flex flex-col items-end gap-6 text-right text-brand-navy/40"
      >
        <Metric n="01" label="Ingest" />
        <Metric n="02" label="Index" />
        <Metric n="03" label="Draft" />
      </div>
    </section>
  );
}

function Metric({ n, label }: { n: string; label: string }) {
  return (
    <div className="leading-none">
      <p className="font-display text-5xl tabular-nums tracking-tight text-brand-navy/15">
        {n}
      </p>
      <p className="mt-1 text-[10px] uppercase tracking-[0.24em] text-brand-navy/40">
        {label}
      </p>
    </div>
  );
}

interface Tile {
  to: string;
  Icon: typeof Upload;
  step: string;
  title: string;
  blurb: string;
  cta: string;
  disabled?: boolean;
}

const TILES: Tile[] = [
  {
    to: "/ingest",
    Icon: Upload,
    step: "01",
    title: "Ingest",
    blurb:
      "Drop a PDF. Docling parses it locally, you confirm the metadata, it lands in the index.",
    cta: "Add a proposal",
  },
  {
    to: "/draft",
    Icon: PenLine,
    step: "02",
    title: "Draft",
    blurb:
      "Pick a section type, point the workflow at a profile, run the critic loop until the draft is yours.",
    cta: "Start drafting",
  },
  {
    to: "/corpus",
    Icon: FileStack,
    step: "03",
    title: "Corpus",
    blurb:
      "Inspect what is indexed locally — proposals, sections, chunks. Read-only browser, v1.",
    cta: "Open corpus",
  },
];

function EntryTiles() {
  return (
    <section
      aria-label="Primary actions"
      className="border-t border-brand-hairline bg-brand-parchment/40 px-6 py-20"
    >
      <div className="mx-auto max-w-5xl">
        <h2 className="font-display text-2xl text-brand-navy">
          Three jobs, no friction.
        </h2>
        <p className="mt-2 max-w-xl text-sm text-brand-navy/65">
          A linear workspace. Bring in the past, draft the present, browse what
          is yours.
        </p>

        <ul className="mt-10 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {TILES.map((tile) => (
            <li key={tile.to}>
              <Tile {...tile} />
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

function Tile({ to, Icon, step, title, blurb, cta }: Tile) {
  return (
    <Link
      to={to}
      className="group relative flex h-full flex-col gap-4 overflow-hidden rounded-xl border border-brand-navy/10 bg-white p-6 shadow-editorial transition-transform hover:-translate-y-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-amber focus-visible:ring-offset-2"
    >
      <span
        aria-hidden="true"
        className="absolute right-5 top-5 font-display text-4xl tabular-nums text-brand-navy/8 transition-colors group-hover:text-brand-amber/50"
        style={{ color: "rgba(10,31,68,0.08)" }}
      >
        {step}
      </span>
      <span
        aria-hidden="true"
        className="inline-flex h-9 w-9 items-center justify-center rounded-md bg-brand-parchment text-brand-navy ring-1 ring-brand-navy/10"
      >
        <Icon className="h-4 w-4" aria-hidden="true" />
      </span>
      <div>
        <h3 className="font-display text-xl text-brand-navy">{title}</h3>
        <p className="mt-2 text-sm leading-relaxed text-brand-navy/70">
          {blurb}
        </p>
      </div>
      <span className="mt-auto inline-flex items-center gap-1 text-sm font-medium text-brand-navy">
        {cta}
        <ArrowUpRight
          className="h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5"
          aria-hidden="true"
        />
      </span>
    </Link>
  );
}

function PrivacyRibbon() {
  return (
    <section
      aria-label="Privacy promise"
      className="border-t border-brand-hairline surface-navy px-6 py-20"
    >
      <div className="mx-auto grid max-w-5xl gap-10 lg:grid-cols-[1fr_2fr]">
        <div>
          <p className="text-xs uppercase tracking-[0.22em] text-brand-amber">
            Privacy
          </p>
          <h2 className="font-display mt-3 text-3xl leading-tight">
            Your proposals never{" "}
            <span className="text-brand-amber">leave the machine.</span>
          </h2>
        </div>
        <div className="space-y-6 text-sm leading-relaxed text-white/80">
          <Promise
            title="Local LLM"
            body="Generation runs through Ollama on your CPU/GPU. No API keys, no third-party endpoints."
          />
          <Promise
            title="Local index"
            body="Chroma persists vectors and metadata under data/. Delete the folder, delete everything."
          />
          <Promise
            title="No telemetry"
            body="EURPE makes no outbound calls outside the optional EU Funding & Tenders auto-fill."
          />
        </div>
      </div>
    </section>
  );
}

function Promise({ title, body }: { title: string; body: string }) {
  return (
    <div className="flex gap-4">
      <span
        aria-hidden="true"
        className="mt-1.5 h-px w-8 shrink-0 bg-brand-amber"
      />
      <div>
        <p className="font-medium text-white">{title}</p>
        <p className="mt-1 text-white/70">{body}</p>
      </div>
    </div>
  );
}

function Footer() {
  return (
    <footer className="border-t border-brand-hairline bg-background px-6 py-10 text-xs text-brand-navy/55">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3">
        <p>
          EURPE — EU Research Proposal Expert.{" "}
          <span className="text-brand-navy/40">Open source · Local-first.</span>
        </p>
        <p className="font-mono">v0.1 · {new Date().getFullYear()}</p>
      </div>
    </footer>
  );
}
