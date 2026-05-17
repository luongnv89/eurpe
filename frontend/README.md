# EURPE frontend

Local-only React UI for EURPE — runs entirely on `127.0.0.1` and never reaches
out to the public internet at runtime.

## Stack

- React 18 + TypeScript
- Vite 5
- Tailwind CSS 3
- shadcn/ui (slate base, CSS variables)

## Quick start

```bash
cd frontend
npm install
npm run dev
```

The dev server starts on `http://127.0.0.1:5173`.

## Layout

```
frontend/
  index.html
  vite.config.ts
  tailwind.config.ts
  components.json          # shadcn/ui config (slate, CSS variables, @/ aliases)
  src/
    main.tsx               # Vite entry
    App.tsx                # Landing screen
    index.css              # Tailwind directives + shadcn CSS variables
    lib/utils.ts           # cn() helper
    components/ui/
      button.tsx           # shadcn Button
```

## Adding more shadcn components

After running `npm install`, add new shadcn primitives via the CLI:

```bash
npx shadcn@latest add card input dialog
```

The `components.json` file already configures the slate base color, CSS
variables, and `@/components` / `@/lib/utils` aliases.

## Build

```bash
npm run build      # tsc -b && vite build
npm run preview    # local production preview on 127.0.0.1:4173
```

## Accessibility

The React UI ships a "best-effort accessibility baseline" in v1 (Task
3.6 / issue #20): keyboard-navigable primary flows with visible focus
states, labelled inputs and generated report regions, WCAG 2.1 AA
contrast on the slate palette, and a deferred-work hand-off for the
v1.2 refactor.

The full audit (numerical contrast ratios, manual tab traces,
verification steps a reviewer can run, and the v1.2 gap list) lives in
[`docs/accessibility.md`](../docs/accessibility.md). Update that file
whenever you change a contrast token, add a new flow, or close one of
the v1.2 deferred items.
