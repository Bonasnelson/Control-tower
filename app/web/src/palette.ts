// Chart colour roles (T1). Validated with the dataviz skill's validate_palette.js on the
// dark surface #15233a (see docs/dashboard_plan.md). Text never wears a series colour.

/** Categorical slots in fixed order (never cycled). At most 3 series per chart. */
export const SERIES = ["#3987e5", "#d95926", "#199e70"] as const;

/** Emphasis form: the highlighted entity takes SERIES[0], everything else is context gray. */
export const EMPHASIS = SERIES[0];
export const CONTEXT = ["#7d8ca8", "#637390"] as const; // 4.64 and 3.29:1 on surface-1

/** Status colours: reserved meaning, always shown with an icon and a text label. */
export const STATUS = {
  critical: "#d03b3b", criticalText: "#f07a7a",
  good: "#0ca30c", goodText: "#3fc46a",
} as const;

/** Chart chrome and ink. */
export const CHART = {
  surface: "#15233a",
  surface2: "#1d2e4a",
  grid: "#2b3d5e",
  axis: "#3a4e72",
  ink: "#eef2f8",
  ink2: "#b4c0d4",
  ink3: "#8f9db5",
  beacon: "#f0b44c",
  fontSize: 12,
} as const;

/** Cities are series: fixed slot per city so a city keeps its colour on every chart. */
export const CITY_COLOR: Record<string, string> = { philadelphia: SERIES[0], chicago: SERIES[1] };
