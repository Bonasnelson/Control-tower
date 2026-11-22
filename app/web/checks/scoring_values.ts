// Run: start the API on :8765, then  node --experimental-strip-types --no-warnings checks/scoring_values.ts
// Recomputes mean ± std (ddof=1) from results/v1_<city>_scoring.csv and compares with the strings the
// Scoring page renders: pm(x, 2) for metrics, pm(x, 1) for score time; plus the chart bar heights / whiskers.
import { readFileSync } from "node:fs";
import { pm } from "../src/api.ts";

let ok = 0, bad = 0;
const eq = (w: string, a: string, b: string) => { if (a === b) ok++; else { bad++; console.log("MISMATCH", w, a, "vs", b); } };
const stats = (xs: number[]) => {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  const sd = xs.length > 1 ? Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1)) : 0;
  return { m, sd };
};
const api = await (await fetch("http://127.0.0.1:8765/api/v1/scoring")).json();
const metrics: [string, number][] = [["auprc", 2], ["recall_at_1pct", 2], ["delta_fp", 2], ["fp_ratio", 2], ["fn_ratio", 2], ["score_seconds", 1]];
for (const city of ["philadelphia", "chicago"]) {
  const [head, ...lines] = readFileSync(`../../results/v1_${city}_scoring.csv`, "utf8").trim().split(/\r?\n/);
  const cols = head.split(",");
  const rows = lines.map((l) => Object.fromEntries(l.split(",").map((v, i) => [cols[i], v])));
  eq(`${city} seeds`, JSON.stringify(api.cities[city].seeds), JSON.stringify([...new Set(rows.map((r) => Number(r.seed)))].sort()));
  for (const scorer of api.scorers) {
    const rs = rows.filter((r) => r.scorer === scorer);
    for (const [m, d] of metrics) {
      const { m: mean, sd } = stats(rs.map((r) => Number(r[m])));
      eq(`${city}/${scorer}/${m} table`, pm(api.cities[city].scorers[scorer][m], d), `${mean.toFixed(d)} ± ${sd.toFixed(d)}`);
      if (m === "auprc" || m === "recall_at_1pct") {
        eq(`${city}/${scorer}/${m} bar`, api.cities[city].scorers[scorer][m].mean.toFixed(6), mean.toFixed(6));
        eq(`${city}/${scorer}/${m} whisker`, api.cities[city].scorers[scorer][m].std.toFixed(6), sd.toFixed(6));
      }
    }
  }
}
console.log(`checked ${ok + bad} values: ${ok} match, ${bad} mismatch`);
