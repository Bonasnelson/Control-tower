// Formatting shared by the Strategies page and checks/strategies_values.ts (plain TS, no JSX).
import type { BufferExp, StrategyRow } from "./api";

export const STRATEGY_METRICS = ["auprc", "recall_at_1pct", "delta_fp", "fn_ratio"] as const;
export type StrategyMetric = (typeof STRATEGY_METRICS)[number];
export const METRIC_LABEL: Record<StrategyMetric, string> = {
  auprc: "AUPRC", recall_at_1pct: "Recall@1%", delta_fp: "ΔFP", fn_ratio: "FN ratio",
};
/** Higher is better for AUPRC and Recall@1%; lower is better for ΔFP and FN ratio. */
export const BETTER: Record<StrategyMetric, "higher" | "lower"> = {
  auprc: "higher", recall_at_1pct: "higher", delta_fp: "lower", fn_ratio: "lower",
};

export const fmt2 = (v: number | null | undefined) => (v === null || v === undefined ? "—" : v.toFixed(2));
export const fmtSeconds = (v: number | null | undefined) => (v === null || v === undefined ? "—" : String(Math.round(v)));
export const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export const strategyCells = (r: StrategyRow) => ({
  label: r.label,
  auprc: fmt2(r.auprc), recall_at_1pct: fmt2(r.recall_at_1pct), delta_fp: fmt2(r.delta_fp),
  fp_ratio: fmt2(r.fp_ratio), fn_ratio: fmt2(r.fn_ratio), train_seconds: fmtSeconds(r.train_seconds),
});

/** The buffer the model replays while training E10 is the state after E9. */
export const usedForE10 = (exps: BufferExp[]) => exps.find((e) => e.experience === "E9");
export const afterE10 = (exps: BufferExp[]) => exps.find((e) => e.experience === "E10");
