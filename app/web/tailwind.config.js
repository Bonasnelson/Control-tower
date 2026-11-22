/** @type {import('tailwindcss').Config} */
// Design tokens (T1). Values mirror the CSS custom properties in src/index.css and
// src/palette.ts; the token table and validator output are in docs/dashboard_plan.md.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Barlow Semi Condensed"', "Barlow", "system-ui", "sans-serif"],
        sans: ['"IBM Plex Sans"', "system-ui", "-apple-system", '"Segoe UI"', "sans-serif"],
        code: ['"IBM Plex Mono"', "ui-monospace", "monospace"],
      },
      fontSize: {
        // 4 steps, ratio ~1.25 (minor third); display sizes use the condensed face
        xs: ["0.75rem", { lineHeight: "1.1rem" }],     // 12 px: axis ticks, footnotes
        sm: ["0.875rem", { lineHeight: "1.35rem" }],   // 14 px: UI, table cells
        base: ["1rem", { lineHeight: "1.6rem" }],      // 16 px: body copy
        lg: ["1.25rem", { lineHeight: "1.6rem" }],     // 20 px: card titles
        xl: ["1.75rem", { lineHeight: "2.1rem" }],     // 28 px: page titles
        figure: ["2.25rem", { lineHeight: "2.4rem" }], // 36 px: headline figures
      },
      colors: {
        plane: "#0f1a2c",      // page background (deep navy, not near-black)
        surface: { 1: "#15233a", 2: "#1d2e4a" }, // panels / raised, drawer, hover
        line: { DEFAULT: "#2b3d5e", strong: "#3a4e72" }, // hairline / axis + baseline
        ink: { DEFAULT: "#eef2f8", 2: "#b4c0d4", 3: "#8f9db5" }, // primary / secondary / muted text
        beacon: "#f0b44c",     // interaction only: focus ring, active nav, links (never data)
        series: { 1: "#3987e5", 2: "#d95926", 3: "#199e70" }, // categorical, fixed order (dataviz dark steps)
        context: { a: "#7d8ca8", b: "#637390" }, // de-emphasised series (emphasis form), both >= 3:1
        critical: { DEFAULT: "#d03b3b", text: "#f07a7a" }, // status: injected anomaly (always icon + label)
        good: { DEFAULT: "#0ca30c", text: "#3fc46a" },     // status: good (always icon + label)
      },
      borderRadius: { panel: "10px", control: "6px" },
    },
  },
  plugins: [],
};
