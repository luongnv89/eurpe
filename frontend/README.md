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
