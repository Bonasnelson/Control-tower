// Formatting shared by the Alerts page and checks/alerts_values.ts (plain TS, no JSX).
import type { Alert, RecordResp } from "./api";

export const KIND_LABEL: Record<string, string> = {
  global_anomaly: "Injected global anomaly",
  local_anomaly: "Injected local anomaly",
  normal: "Not injected",
};

/** Scores differ in scale per scorer (surprisal ~1–30, isolation forest ~0.3–0.8): 3 significant decimals. */
export const formatScore = (v: number) => v.toFixed(3);

/** Group of a record: its department, or "injected" for records that belong to no department group
 *  (their own dept field still has a value; that is shown in the field table). */
export const departmentLabel = (d: string | null | undefined) => (d ? d : "Injected (not in a department group)");
export const groupShort = (d: string | null | undefined) => (d ? d : "injected");

export const alertRowText = (a: Alert) => ({
  rank: String(a.rank),
  record: String(a.row),
  department: departmentLabel(a.department),
  score: formatScore(a.score),
  injected: KIND_LABEL[a.kind] ?? a.kind,
  reasons: a.reason_codes.join(", "),
});

/** Per-field surprisal, highest first (the drawer's bar order). */
export const sortedFields = (r: RecordResp) => [...r.fields].sort((a, b) => b.surprisal - a.surprisal);

export const formatValue = (v: string | number) => (typeof v === "number" ? String(v) : v);
export const formatSurprisal = (v: number) => v.toFixed(2);
