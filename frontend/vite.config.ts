import path from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// EURPE local dev server. Bind to 127.0.0.1 — never expose the UI to a public
// interface; this is a local-only application.
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
  },
  preview: {
    host: "127.0.0.1",
    port: 4173,
  },
});
