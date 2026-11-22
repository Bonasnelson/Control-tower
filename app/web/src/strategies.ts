// Formatting shared by the Strategies page and checks/strategies_values.ts (plain TS, no JSX).
import type { BufferExp, MeanStd, StrategyRow } from "./api";

export const STRATEGY_METRICS = ["auprc", "recall_at_1pct", "delta_fp", "fn_ratio"] as const;
export type StrategyMetric = (typeof STRATEGY_METRICS)[number];
export const METRIC_LABEL: Record<StrategyMetric, string> = {
  auprc: "AUPRC", recall_at_1pct: "Recall@1%", delta_fp: "ΔFP", fn_ratio: "FN ratio",
};
/** Higher is better for AUPRC and Recall@1%; lower is better for ΔFP and FN ratio. */
export const BETTER: Record<StrategyMetric, "higher" | "lower"> = {
  auprc: "higher", recall_at_1pct: "higher", delta_fp: "lower", fn_ratio: "lower",
};

const num = (v: number | null | undefined, d: number) => (v === null || v === undefined ? "—" : v.toFixed(d));
/** mean ± std (std over seeds, ddof = 1); a single seed shows the value alone. */
export const fmtMS = (m: MeanStd | undefined, d = 2) =>
  !m || m.mean === null ? "—" : (m.n ?? 1) > 1 ? `${num(m.mean, d)} ± ${num(m.std, d)}` : num(m.mean, d);
export const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)}%`);
export const pctMS = (m: MeanStd | undefined) =>
  !m || m.mean === null ? "—" : (m.n ?? 1) > 1 ? `${pct(m.mean)} ± ${pct(m.std)}` : pct(m.mean);

export const strategyCells = (r: StrategyRow) => ({
  label: r.label,
  auprc: fmtMS(r.auprc), recall_at_1pct: fmtMS(r.recall_at_1pct), delta_fp: fmtMS(r.delta_fp),
  fp_ratio: fmtMS(r.fp_ratio), fn_ratio: fmtMS(r.fn_ratio), train_seconds: fmtMS(r.train_seconds, 0),
});

/** The buffer the model replays while training E10 is the state after E9. */
export const usedForE10 = (exps: BufferExp[]) => exps.find((e) => e.experience === "E9");
export const afterE10 = (exps: BufferExp[]) => exps.find((e) => e.experience === "E10");
