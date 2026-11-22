// Run: start the API on :8765, then  node --experimental-strip-types --no-warnings checks/stream_values.ts
// Compares what the Stream page renders (API values formatted with the page's Intl.NumberFormat)
// with the raw cells of results/experiences_<city>_linear.csv, and checks the headline figures.
import { readFileSync } from "node:fs";

const nf = new Intl.NumberFormat("en-US");
let ok = 0, bad = 0;
const eq = (what: string, a: string, b: string) => { if (a === b) ok++; else { bad++; console.log("MISMATCH", what, a, "vs", b); } };

for (const city of ["philadelphia", "chicago"]) {
  const api = await (await fetch(`http://127.0.0.1:8765/api/experiences/${city}`)).json();
  // the CSV's department names contain no commas, so a plain split is safe here
  const [head, ...lines] = readFileSync(`../../results/experiences_${city}_linear.csv`, "utf8").trim().split(/\r?\n/);
  const cols = head.split(",");
  eq(`${city} departments`, JSON.stringify(api.departments), JSON.stringify(cols.slice(1, -1)));
  lines.forEach((line, i) => {
    const v = line.split(",");
    const e = api.experiences[i];
    eq(`${city} row label`, e.experience, v[0]);
    cols.slice(1).forEach((c, j) => eq(`${city} ${v[0]} ${c}`, nf.format(Number(e[c])), nf.format(Number(v[j + 1]))));
  });
  const t = api.target_dept, first = lines[0].split(","), last = lines[lines.length - 1].split(",");
  const ti = cols.indexOf(t);
  eq(`${city} target E1`, nf.format(Number(api.experiences[0][t])), nf.format(Number(first[ti])));
  eq(`${city} target E10`, nf.format(Number(api.experiences.at(-1)[t])), nf.format(Number(last[ti])));
  console.log(`${city}: target ${t}, E1 ${first[ti]} -> E10 ${last[ti]}, E10 total ${last.at(-1)}`);
}
console.log(`checked ${ok + bad} cells: ${ok} match, ${bad} mismatch`);
