import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import { AppShell } from "@/components/AppShell";
import { CorpusBrowser } from "@/features/corpus/CorpusBrowser";
import { DraftingWorkspace } from "@/features/drafting/DraftingWorkspace";
import { HomePage } from "@/features/home/HomePage";
import { IngestWizard } from "@/features/ingest/IngestWizard";
import { SettingsPage } from "@/features/settings/SettingsPage";

/**
 * Top-level router. The persistent AppShell renders the sidebar + skip
 * link and an Outlet for the active route. Routes:
 *
 *   /          Home (editorial landing)
 *   /ingest    Three-step proposal ingestion wizard
 *   /draft     Section drafting workspace + critic loop
 *   /corpus    Read-only browser (v1.1 placeholder)
 *   /settings  Effective config summary (read-only)
 *
 * All routes share the AppShell so navigation, the offline pill, and
 * the skip link stay consistent across pages.
 */
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<HomePage />} />
          <Route path="ingest" element={<IngestWizard />} />
          <Route path="draft" element={<DraftingWorkspace />} />
          <Route path="corpus" element={<CorpusBrowser />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
