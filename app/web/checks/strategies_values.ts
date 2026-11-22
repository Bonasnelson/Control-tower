// Run: start the API on :8765, then  node --experimental-strip-types --no-warnings checks/strategies_values.ts
// Renders the Strategies tables with the page's own helpers (src/strategies.ts) from the API and compares
// them with results/strategies_<city>.csv and results/dashboard/<city>_buffer.json read from disk.
import { readFileSync } from "node:fs";
import { afterE10, fmt2, fmtSeconds, pct, strategyCells, usedForE10 } from "../src/strategies.ts";

let ok = 0, bad = 0;
const eq = (w: string, a: string, b: string) => { if (a === b) ok++; else { bad++; console.log("MISMATCH", w, a, "vs", b); } };
const api = await (await fetch("http://127.0.0.1:8765/api/strategies")).json();
const ANOM = ["GLOBAL_ANOM", "LOCAL_ANOM"];

for (const city of ["philadelphia", "chicago"]) {
  const [head, ...lines] = readFileSync(`../../results/strategies_${city}.csv`, "utf8").trim().split(/\r?\n/);
  const cols = head.split(",");
  const rows = lines.map((l) => Object.fromEntries(l.split(",").map((v, i) => [cols[i], v])));
  const c = api.cities[city];
  eq(`${city} row count`, String(c.rows.length), String(rows.length));
  eq(`${city} seeds`, JSON.stringify(c.seeds), JSON.stringify([...new Set(rows.map((r) => Number(r.seed)))]));
  rows.forEach((r, i) => {
    const cells = strategyCells(c.rows[i]);
    const k = `${city}/${r.strategy}/${r.buffer_rule || "-"}`;
    eq(`${k} identity`, `${c.rows[i].strategy}/${c.rows[i].buffer_rule ?? ""}`, `${r.strategy}/${r.buffer_rule}`);
    for (const m of ["auprc", "recall_at_1pct", "delta_fp", "fp_ratio", "fn_ratio"] as const) eq(`${k} ${m}`, cells[m], fmt2(Number(r[m])));
    eq(`${k} train_seconds`, cells.train_seconds, fmtSeconds(Number(r.train_seconds)));
  });

  const buf = JSON.parse(readFileSync(`../../results/dashboard/${city}_buffer.json`, "utf8"));
  eq(`${city} buffer target`, c.buffer.target_department, buf.target_department);
  buf.runs.forEach((run: any, i: number) => {
    const ar = c.buffer.runs[i];
    eq(`${city} run ${i} identity`, `${ar.strategy}/${ar.buffer_rule}`, `${run.strategy}/${run.buffer_rule}`);
    run.experiences.forEach((e: any, j: number) => {
      const n = Object.values<number>(e.counts).reduce((a, b) => a + b, 0);
      const t = e.counts[buf.target_department] ?? 0;
      const inj = ANOM.reduce((a, k) => a + (e.counts[k] ?? 0), 0);
      eq(`${city} ${ar.label} ${e.experience} target share`, pct(ar.experiences[j].target_share), pct(t / n));
      eq(`${city} ${ar.label} ${e.experience} injected`, String(ar.experiences[j].injected), String(inj));
      eq(`${city} ${ar.label} ${e.experience} size`, String(ar.experiences[j].size), String(e.size));
    });
    const e9 = usedForE10(ar.experiences)!, e10 = afterE10(ar.experiences)!;
    eq(`${city} ${ar.label} E9 has no injected records`, String(e9.injected), "0");
    console.log(`${city} ${ar.label}: E9 target ${e9.target}/${e9.size} (${pct(e9.target_share)}), E10 injected ${e10.injected}`);
  });
}
console.log(`checked ${ok + bad} values: ${ok} match, ${bad} mismatch`);
