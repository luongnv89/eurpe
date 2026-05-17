import type { Config } from "tailwindcss";

// EURPE Tailwind config — shadcn/ui compatible (slate base color, CSS variables).
const config: Config = {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx,js,jsx}",
  ],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        // EURPE × Montimage blended palette. Navy anchors structure/surfaces;
        // Amber is the single accent (CTAs, focus, active states, hairlines).
        // Amber #E9AB34 fails AA as text on white (~2.5:1), so use it ONLY:
        //   - as a fill behind navy text (passes 6:1), or
        //   - as a non-text mark (underline, dot, rule, focus ring).
        // Gold (#FFD617) is retained as a logo-internal token only — do not
        // use in app surfaces.
        brand: {
          navy: "#0A1F44",
          "navy-700": "#162E5B",
          "navy-500": "#23437A",
          amber: "#E9AB34",
          "amber-600": "#C68A1F",
          "amber-100": "#FBE7B8",
          paper: "#FAFAFA",
          parchment: "#F5F0EB",
          ink: "#111827",
          muted: "#6B7280",
          hairline: "#E5E0D6",
          gold: "#FFD617", // legacy / logo-internal only
        },
      },
      fontFamily: {
        serif: [
          '"Source Serif 4"',
          '"Source Serif Pro"',
          'Charter',
          'Georgia',
          'serif',
        ],
        sans: [
          '"Inter"',
          'ui-sans-serif',
          'system-ui',
          '-apple-system',
          '"Segoe UI"',
          'Roboto',
          'sans-serif',
        ],
        mono: [
          '"JetBrains Mono"',
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'monospace',
        ],
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      boxShadow: {
        editorial: "0 1px 0 0 rgba(10,31,68,0.04), 0 12px 24px -16px rgba(10,31,68,0.18)",
        amber: "0 1px 0 0 rgba(198,138,31,0.4), 0 8px 18px -10px rgba(233,171,52,0.45)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};

export default config;
