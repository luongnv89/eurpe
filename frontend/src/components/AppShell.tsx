import { useState } from "react";

import { NavLink, Outlet, useLocation } from "react-router-dom";
import {
  FileStack,
  LayoutDashboard,
  Menu,
  PenLine,
  Settings,
  ShieldCheck,
  Upload,
  X,
} from "lucide-react";

import { cn } from "@/lib/utils";

interface NavItem {
  to: string;
  label: string;
  Icon: typeof LayoutDashboard;
  end?: boolean;
}

const NAV: NavItem[] = [
  { to: "/", label: "Home", Icon: LayoutDashboard, end: true },
  { to: "/ingest", label: "Ingest", Icon: Upload },
  { to: "/draft", label: "Draft", Icon: PenLine },
  { to: "/corpus", label: "Corpus", Icon: FileStack },
  { to: "/settings", label: "Settings", Icon: Settings },
];

/**
 * Persistent navigation shell. Sidebar (lg+) collapses to a top drawer on
 * mobile. The skip-link is the first focusable element so keyboard users
 * always have a one-shot path to the main content.
 */
export function AppShell() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const location = useLocation();
  const pageLabel =
    NAV.find((n) => (n.end ? location.pathname === n.to : location.pathname.startsWith(n.to)))?.label ??
    "Home";

  return (
    <div className="min-h-screen bg-background text-foreground">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-brand-amber focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:text-brand-navy focus:shadow-md focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
      >
        Skip to main content
      </a>

      {/* Desktop sidebar */}
      <aside
        aria-label="Primary navigation"
        className="hidden lg:flex fixed inset-y-0 left-0 w-64 flex-col surface-navy"
      >
        <Brand />
        <NavList onNavigate={() => setDrawerOpen(false)} />
        <OfflinePill className="mx-4 mb-6 mt-auto" />
      </aside>

      {/* Mobile top bar */}
      <header className="lg:hidden flex items-center justify-between border-b border-brand-hairline bg-background/90 px-4 py-3 backdrop-blur">
        <button
          type="button"
          onClick={() => setDrawerOpen(true)}
          className="inline-flex items-center gap-2 rounded-md border border-brand-navy/15 px-3 py-2 text-sm font-medium text-brand-navy hover:bg-brand-parchment focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          aria-label="Open navigation menu"
        >
          <Menu className="h-4 w-4" aria-hidden="true" />
          <span>{pageLabel}</span>
        </button>
        <BrandWordmark className="text-brand-navy" />
      </header>

      {/* Mobile drawer */}
      {drawerOpen && (
        <div
          className="lg:hidden fixed inset-0 z-40"
          role="dialog"
          aria-modal="true"
          aria-label="Navigation"
        >
          <div
            className="absolute inset-0 bg-brand-navy/40"
            onClick={() => setDrawerOpen(false)}
            aria-hidden="true"
          />
          <div className="relative h-full w-72 surface-navy flex flex-col">
            <div className="flex items-center justify-between px-6 pt-6">
              <BrandWordmark />
              <button
                type="button"
                onClick={() => setDrawerOpen(false)}
                className="rounded-md p-1 text-white/80 hover:bg-white/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-amber"
                aria-label="Close navigation"
              >
                <X className="h-5 w-5" aria-hidden="true" />
              </button>
            </div>
            <NavList onNavigate={() => setDrawerOpen(false)} />
            <OfflinePill className="mx-4 mb-6 mt-auto" />
          </div>
        </div>
      )}

      <main
        id="main-content"
        tabIndex={-1}
        className="lg:pl-64 outline-none"
      >
        <Outlet />
      </main>
    </div>
  );
}

function Brand() {
  return (
    <div className="flex items-center gap-3 px-6 pt-8 pb-10">
      <span
        aria-hidden="true"
        className="grid h-9 w-9 place-items-center rounded-md bg-brand-amber font-display text-lg font-semibold text-brand-navy"
      >
        E
      </span>
      <div className="leading-tight">
        <p className="font-display text-lg text-white">EURPE</p>
        <p className="text-[11px] uppercase tracking-[0.18em] text-brand-amber-100/80">
          Proposal Expert
        </p>
      </div>
    </div>
  );
}

function BrandWordmark({ className }: { className?: string }) {
  return (
    <span className={cn("inline-flex items-center gap-2", className)}>
      <span
        aria-hidden="true"
        className="grid h-7 w-7 place-items-center rounded bg-brand-amber font-display text-sm font-semibold text-brand-navy"
      >
        E
      </span>
      <span className="font-display text-base">EURPE</span>
    </span>
  );
}

function NavList({ onNavigate }: { onNavigate: () => void }) {
  return (
    <nav className="px-3" aria-label="Primary">
      <ul className="space-y-0.5">
        {NAV.map(({ to, label, Icon, end }) => (
          <li key={to}>
            <NavLink
              to={to}
              end={end}
              onClick={onNavigate}
              className={({ isActive }) =>
                cn(
                  "group flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  "text-white/80 hover:bg-white/5 hover:text-white",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-amber focus-visible:ring-offset-2 focus-visible:ring-offset-brand-navy",
                  isActive && "bg-white/10 text-white",
                )
              }
            >
              {({ isActive }) => (
                <>
                  <span
                    aria-hidden="true"
                    className={cn(
                      "h-5 w-0.5 rounded-full transition-colors",
                      isActive ? "bg-brand-amber" : "bg-transparent",
                    )}
                  />
                  <Icon className="h-4 w-4" aria-hidden="true" />
                  <span>{label}</span>
                </>
              )}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}

function OfflinePill({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "rounded-md border border-white/10 bg-white/[0.04] px-3 py-2 text-xs text-white/80",
        className,
      )}
    >
      <span className="inline-flex items-center gap-2">
        <ShieldCheck className="h-3.5 w-3.5 text-brand-amber" aria-hidden="true" />
        <span>Local-only · Offline by default</span>
      </span>
    </div>
  );
}
