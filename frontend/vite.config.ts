import path from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// EURPE local dev server. Bind to 127.0.0.1 — never expose the UI to a public
// interface; this is a local-only application.
//
// The ``server.proxy`` entry below forwards all ``/api/*`` calls to the FastAPI
// service running on 127.0.0.1:8765. The frontend uses relative URLs (see
// ``src/features/ingest/api.ts``) so the same code path works in dev (proxied
// here) and in production (served alongside the FastAPI app).
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: false,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: false,
      },
    },
  },
  preview: {
    host: "127.0.0.1",
    port: 4173,
  },
});
