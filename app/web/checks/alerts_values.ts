// Run: start the API on :8765, then  node --experimental-strip-types --no-warnings checks/alerts_values.ts
// Renders every alert row and every drawer record with the page's own helpers (src/alerts.ts) from the
// API, and compares them with results/dashboard/<city>_alerts.json read directly from disk.
import { readFileSync } from "node:fs";
import { KIND_LABEL, alertRowText, departmentLabel, formatScore, formatSurprisal, formatValue, groupShort, sortedFields } from "../src/alerts.ts";

let ok = 0, bad = 0;
const eq = (w: string, a: string, b: string) => { if (a === b) ok++; else { bad++; console.log("MISMATCH", w, a, "vs", b); } };
let rowsChecked = 0, recordsChecked = 0;
for (const city of ["philadelphia", "chicago"]) {
  const file = JSON.parse(readFileSync(`../../results/dashboard/${city}_alerts.json`, "utf8"));
  for (const scorer of Object.keys(file.alerts)) {
    const api = await (await fetch(`http://127.0.0.1:8765/api/v1/alerts/${city}?scorer=${scorer}`)).json();
    eq(`${city}/${scorer} n_alerts`, String(api.alerts.length), String(file.n_alerts));
    file.alerts[scorer].forEach((fa: any, i: number) => {
      const t = alertRowText(api.alerts[i]);
      const rec = file.records[String(fa.row)];
      const k = `${city}/${scorer}#${fa.rank}`;
      eq(`${k} rank`, t.rank, String(fa.rank)); eq(`${k} row`, t.record, String(fa.row));
      eq(`${k} dept`, t.department, departmentLabel(rec.department)); eq(`${k} score`, t.score, formatScore(fa.score));
      eq(`${k} injected`, t.injected, KIND_LABEL[fa.kind]); eq(`${k} reasons`, t.reasons, rec.reason_codes.join(", "));
      rowsChecked++;
    });
  }
  for (const [row, fr] of Object.entries<any>(file.records)) {
    const r = await (await fetch(`http://127.0.0.1:8765/api/v1/record/${city}/${row}`)).json();
    const k = `${city}/record ${row}`;
    eq(`${k} kind`, r.kind, fr.kind);
    eq(`${k} drawer group`, groupShort(r.department), fr.department ?? "injected");
    if (fr.kind !== "normal") eq(`${k} injected => no group`, String(fr.department), "null"); eq(`${k} reasons`, r.reason_codes.join(","), fr.reason_codes.join(","));
    eq(`${k} bar order`, sortedFields(r).map((f: any) => f.name).join(","),
      [...fr.fields].sort((a: any, b: any) => b.surprisal - a.surprisal).map((f: any) => f.name).join(","));
    fr.fields.forEach((f: any, i: number) => {
      eq(`${k} ${f.name} value`, formatValue(r.fields[i].value), formatValue(f.value));
      eq(`${k} ${f.name} surprisal`, formatSurprisal(r.fields[i].surprisal), formatSurprisal(f.surprisal));
    });
    // top-3 reason codes are the three highest-surprisal fields
    eq(`${k} reasons = top-3 surprisal`, fr.reason_codes.join(","),
      [...fr.fields].sort((a: any, b: any) => b.surprisal - a.surprisal).slice(0, 3).map((f: any) => f.name).join(","));
    recordsChecked++;
  }
}
console.log(`alert rows ${rowsChecked}, drawer records ${recordsChecked}; checked ${ok + bad} values: ${ok} match, ${bad} mismatch`);
