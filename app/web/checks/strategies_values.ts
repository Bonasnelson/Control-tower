// Run: start the API on :8765, then  node --experimental-strip-types --no-warnings checks/strategies_values.ts
// Recomputes mean ± std over seeds (ddof = 1) per configuration from results/strategies_<city>.csv and
// results/dashboard/<city>_buffer.json, and compares with the strings the Strategies page renders
// (src/strategies.ts helpers applied to the API response).
import { readFileSync } from "node:fs";
import { afterE10, fmtMS, pctMS, strategyCells, usedForE10 } from "../src/strategies.ts";  // pctMS: log line

let ok = 0, bad = 0;
const eq = (w: string, a: string, b: string) => { if (a === b) ok++; else { bad++; console.log("MISMATCH", w, a, "vs", b); } };
const ms = (xs: number[]) => {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  const sd = xs.length > 1 ? Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1)) : 0;
  return { mean: m, std: sd, n: xs.length };
};
const api = await (await fetch("http://127.0.0.1:8765/api/strategies")).json();
const ANOM = ["GLOBAL_ANOM", "LOCAL_ANOM"];

for (const city of ["philadelphia", "chicago"]) {
  const [head, ...lines] = readFileSync(`../../results/strategies_${city}.csv`, "utf8").trim().split(/\r?\n/);
  const cols = head.split(",");
  const rows = lines.map((l) => Object.fromEntries(l.split(",").map((v, i) => [cols[i], v])));
  const c = api.cities[city];
  const keys = [...new Set(rows.map((r) => `${r.strategy}/${r.buffer_rule}`))];
  eq(`${city} configurations`, String(c.rows.length), String(keys.length));
  eq(`${city} seeds`, JSON.stringify(c.seeds), JSON.stringify([...new Set(rows.map((r) => Number(r.seed)))].sort()));
  keys.forEach((k, i) => {
    const rs = rows.filter((r) => `${r.strategy}/${r.buffer_rule}` === k);
    const cells = strategyCells(c.rows[i]);
    eq(`${city} ${k} identity`, `${c.rows[i].strategy}/${c.rows[i].buffer_rule ?? ""}`, k);
    eq(`${city} ${k} seeds`, JSON.stringify(c.rows[i].seeds), JSON.stringify(rs.map((r) => Number(r.seed)).sort()));
    for (const m of ["auprc", "recall_at_1pct", "delta_fp", "fp_ratio", "fn_ratio"] as const)
      eq(`${city} ${k} ${m}`, cells[m], fmtMS(ms(rs.map((r) => Number(r[m])))));
    eq(`${city} ${k} train_seconds`, cells.train_seconds, fmtMS(ms(rs.map((r) => Number(r.train_seconds))), 0));
    // chart bar = mean, whisker = std
    eq(`${city} ${k} bar`, c.rows[i].auprc.mean.toFixed(6), ms(rs.map((r) => Number(r.auprc))).mean.toFixed(6));
    eq(`${city} ${k} whisker`, c.rows[i].auprc.std.toFixed(6), ms(rs.map((r) => Number(r.auprc))).std.toFixed(6));
  });

  const buf = JSON.parse(readFileSync(`../../results/dashboard/${city}_buffer.json`, "utf8"));
  const bkeys = [...new Set(buf.runs.map((r: any) => `${r.strategy}/${r.buffer_rule}`))] as string[];
  bkeys.forEach((k, i) => {
    const runs = buf.runs.filter((r: any) => `${r.strategy}/${r.buffer_rule}` === k);
    const ar = c.buffer.runs[i];
    eq(`${city} buffer ${k} identity`, `${ar.strategy}/${ar.buffer_rule}`, k);
    for (const e of ar.experiences) {
      const entries = runs.map((r: any) => r.experiences.find((x: any) => x.experience === e.experience));
      const share = entries.map((x: any) => (x.counts[buf.target_department] ?? 0) / Object.values<number>(x.counts).reduce((a, b) => a + b, 0));
      const inj = entries.map((x: any) => ANOM.reduce((a, g) => a + (x.counts[g] ?? 0), 0));
      // numeric comparison: shares on a rounding boundary (e.g. 9.75%) differ in the last float bit
      // between Python's exact mean and a JS sum, which can flip the 1-decimal display
      const ref = ms(share);
      eq(`${city} ${k} ${e.experience} target share`, String(Math.abs(e.target_share.mean - ref.mean) < 1e-9 && Math.abs(e.target_share.std - ref.std) < 1e-9), "true");
      eq(`${city} ${k} ${e.experience} injected`, fmtMS(e.injected, 1), fmtMS(ms(inj), 1));
    }
    const e9 = usedForE10(ar.experiences)!, e10 = afterE10(ar.experiences)!;
    eq(`${city} ${k} E9 has no injected records`, String(e9.injected.mean), "0");
    console.log(`${city} ${ar.label}: seeds ${ar.seeds.join(",")}, E9 target ${pctMS(e9.target_share)}, E10 injected ${fmtMS(e10.injected, 1)}`);
  });
}
console.log(`checked ${ok + bad} values: ${ok} match, ${bad} mismatch`);
