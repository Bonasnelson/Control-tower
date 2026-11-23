// Run: start the API on :8765, then  node --experimental-strip-types --no-warnings checks/spread_values.ts
// Separate scenario (linear_spread, anomalies injected in E5-E10): recomputes mean ± std over seeds (ddof = 1)
// per configuration from results/strategies_<city>_linear_spread.csv and compares with the strings the
// Strategies page renders (spreadCells) and with the chart bars / whiskers. Also checks that the main
// /api/strategies endpoint does not include the linear_spread files.
import { readFileSync } from "node:fs";
import { fmtMS, spreadCells } from "../src/strategies.ts";

let ok = 0, bad = 0;
const eq = (w: string, a: string, b: string) => { if (a === b) ok++; else { bad++; console.log("MISMATCH", w, a, "vs", b); } };
const ms = (xs: number[]) => {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  const sd = xs.length > 1 ? Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1)) : 0;
  return { mean: m, std: sd, n: xs.length };
};
const api = await (await fetch("http://127.0.0.1:8765/api/strategies/linear_spread")).json();
const main = await (await fetch("http://127.0.0.1:8765/api/strategies")).json();
eq("scenario", api.scenario, "linear_spread");

for (const city of ["philadelphia", "chicago"]) {
  eq(`${city} main endpoint source is the linear file`, main.cities[city].source, `results/strategies_${city}.csv`);
  eq(`${city} main endpoint has no spread field`, String("spread" in main.cities[city]), "false");
  const [head, ...lines] = readFileSync(`../../results/strategies_${city}_linear_spread.csv`, "utf8").trim().split(/\r?\n/);
  const cols = head.split(",");
  const rows = lines.map((l) => Object.fromEntries(l.split(",").map((v, i) => [cols[i], v])));
  const c = api.cities[city];
  eq(`${city} source`, c.source, `results/strategies_${city}_linear_spread.csv`);
  const keys = [...new Set(rows.map((r) => `${r.strategy}/${r.buffer_rule}`))];
  eq(`${city} configurations`, String(c.rows.length), String(keys.length));
  keys.forEach((k, i) => {
    const rs = rows.filter((r) => `${r.strategy}/${r.buffer_rule}` === k);
    const row = c.rows[i], cells = spreadCells(row);
    eq(`${city} ${k} identity`, `${row.strategy}/${row.buffer_rule ?? ""}`, k);
    eq(`${city} ${k} seeds`, JSON.stringify(row.seeds), JSON.stringify(rs.map((r) => Number(r.seed)).sort()));
    for (const m of ["auprc", "recall_at_1pct", "delta_fp", "fn_ratio"] as const)
      eq(`${city} ${k} ${m}`, cells[m], fmtMS(ms(rs.map((r) => Number(r[m])))));
    const opt = (col: string, d: number) =>
      rs.every((r) => r[col] === "" || r[col] === undefined) ? "—" : fmtMS(ms(rs.map((r) => Number(r[col]))), d);
    eq(`${city} ${k} used buffers`, cells.used, opt("buffer_injected_used_total", 1));
    eq(`${city} ${k} after E9`, cells.e9, opt("buffer_injected_after_E9", 1));
    eq(`${city} ${k} dropped`, cells.dropped, opt("filter_dropped_injected_total", 1));
    if (row.buffer_injected_used_total) {                    // chart bar and whisker
      const ref = ms(rs.map((r) => Number(r.buffer_injected_used_total)));
      eq(`${city} ${k} bar`, row.buffer_injected_used_total.mean.toFixed(6), ref.mean.toFixed(6));
      eq(`${city} ${k} whisker`, row.buffer_injected_used_total.std.toFixed(6), ref.std.toFixed(6));
    }
    console.log(`${city} ${row.label}: AUPRC ${cells.auprc}, used ${cells.used}, after E9 ${cells.e9}, dropped ${cells.dropped}`);
  });
}
console.log(`checked ${ok + bad} values: ${ok} match, ${bad} mismatch`);
