import type { BaselinesResp } from "./api";

const ORDER = ["linear", "exponential", "instant"];
const SCEN_SHORT: Record<string, string> = { linear: "linear", exponential: "exp.", instant: "instant" };
/** SEL trains only on E10, which every decay scenario shares, so its rows are identical:
 *  show one row. Falls back to the separate rows if they ever differ. */
export function collapseSel(rows: BaselinesResp["rows"]): BaselinesResp["rows"] {
  const sel = rows.filter((r) => r.regime === "SEL");
  const key = (r: BaselinesResp["rows"][number]) =>
    JSON.stringify([r.seeds, r.delta_fp_x1e3, r.fp_ratio, r.fn_ratio, r.auprc, r.recall_at_1pct]);
  if (sel.length < 2 || sel.some((r) => key(r) !== key(sel[0]))) return rows;
  const merged = { ...sel[0], scenario: `any decay (${sel.map((r) => r.scenario).sort((x, y) => ORDER.indexOf(x) - ORDER.indexOf(y)).map((x) => SCEN_SHORT[x] ?? x).join(" / ")})` };
  return [...rows.filter((r) => r.regime !== "SEL"), merged];
}
