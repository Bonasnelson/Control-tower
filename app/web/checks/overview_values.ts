// Run: start the API on :8765, then  node --experimental-strip-types --no-warnings checks/overview_values.ts
// Renders the strings the Overview page shows (same pm() calls) and compares them with
// (a) the baseline summary markdown cells (3 dp, as printed there),
// (b) v1 values recomputed here from results/v1_<city>_scoring.csv (mean, std ddof=1, 2 dp),
// (c) the README "Week 2 result" table strings.
import { readFileSync } from "node:fs";
import { pm } from "../src/api.ts";
import { collapseSel } from "../src/baselines.ts";

const root = "../..";
const read = (p: string) => readFileSync(`${root}/${p}`, "utf8");
const cells = (line: string) => line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
let bad = 0, ok = 0;
const eq = (what: string, page: string, ref: string) => {
  if (page === ref) ok++;
  else { bad++; console.log("MISMATCH", what, "page", JSON.stringify(page), "ref", JSON.stringify(ref)); }
};

// (a) baselines: the page shows collapseSel(rows); SEL is merged only if the .md SEL rows are identical
const base = await (await fetch("http://127.0.0.1:8765/api/baselines")).json();
const pageRows = collapseSel(base.rows);
const mdRows = read("results/summary_philadelphia_baseline_ae.md").split(/\r?\n/)
  .filter((l) => l.startsWith("| ") && !l.startsWith("| regime")).map(cells);
const selMd = mdRows.filter((c) => c[0] === "SEL");
const selIdentical = selMd.every((c) => c.slice(2).join("|") === selMd[0].slice(2).join("|"));
eq("SEL rows identical in .md => merged on page", String(pageRows.filter((r: any) => r.regime === "SEL").length), selIdentical ? "1" : String(selMd.length));
console.log(`SEL rows in .md identical: ${selIdentical}; page rows: ${pageRows.map((r: any) => `${r.regime} ${r.scenario}`).join(", ")}`);
for (const c of mdRows) {
  const r = pageRows.find((x: any) => x.regime === c[0] && (x.scenario === c[1] || (selIdentical && c[0] === "SEL")));
  const k = `${c[0]}/${c[1]}`;
  eq(`${k} dFP`, pm(r.delta_fp_x1e3), c[3]); eq(`${k} FP`, pm(r.fp_ratio), c[4]);
  eq(`${k} FN`, pm(r.fn_ratio), c[5]); eq(`${k} AUPRC`, pm(r.auprc), c[6]); eq(`${k} R@1%`, pm(r.recall_at_1pct), c[7]);
}
eq("paper SEL string vs README", base.paper.SEL.delta_fp_x1e3, "+1.45 / +1.40 / +1.45");

// (b) v1 from the CSVs
const v1 = await (await fetch("http://127.0.0.1:8765/api/v1/scoring")).json();
const meanStd = (xs: number[]) => {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  const sd = xs.length > 1 ? Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1)) : 0;
  return `${m.toFixed(2)} ± ${sd.toFixed(2)}`;
};
const page: Record<string, string> = {};
for (const city of ["philadelphia", "chicago"]) {
  const [head, ...lines] = read(`results/v1_${city}_scoring.csv`).trim().split(/\r?\n/);
  const cols = head.split(",");
  const rows = lines.map((l) => Object.fromEntries(l.split(",").map((v, i) => [cols[i], v])));
  for (const scorer of v1.scorers) {
    const rs = rows.filter((r) => r.scorer === scorer);
    for (const m of ["auprc", "recall_at_1pct"]) {
      const p = pm(v1.cities[city].scorers[scorer][m], 2);
      page[`${city}/${scorer}/${m}`] = p;
      eq(`${city}/${scorer} ${m} vs CSV`, p, meanStd(rs.map((r) => Number(r[m]))));
    }
  }
}

// (c) README Week-2 table
const LABEL: Record<string, string> = { "Reconstruction error": "recon_error", "Field surprisal": "field_surprisal",
  "Latent k-NN distance": "latent_distance", "Isolation forest": "isolation_forest" };
const readme = read("README.md");
const sec = readme.slice(readme.indexOf("### Week 2 result: model v1 scoring comparison"));
for (const line of sec.split("\n").filter((l) => l.startsWith("| ") && !l.startsWith("| Scorer")).slice(0, 4)) {
  const c = cells(line);
  const s = LABEL[c[0]];
  eq(`README ${c[0]} Phila AUPRC`, page[`philadelphia/${s}/auprc`], c[1]);
  eq(`README ${c[0]} Phila R@1%`, page[`philadelphia/${s}/recall_at_1pct`], c[2]);
  eq(`README ${c[0]} Chicago AUPRC`, page[`chicago/${s}/auprc`], c[3]);
  eq(`README ${c[0]} Chicago R@1%`, page[`chicago/${s}/recall_at_1pct`], c[4]);
}
console.log(`checked ${ok + bad} cells: ${ok} match, ${bad} mismatch`);
