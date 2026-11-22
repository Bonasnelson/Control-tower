import { useCallback, useState } from "react";
import type { AlertsResp, RecordResp } from "../api";
import { SCORER_LABEL, isPending, useApi } from "../api";
import { alertRowText, formatSurprisal, formatValue, groupShort, sortedFields } from "../alerts";
import { DataTable, Drawer, Figure, InjectionLabel, Loader, PageHeader, Panel, Segmented, Source } from "../components/ui";
import { CONTEXT } from "../palette";

const CITY_NAME: Record<string, string> = { philadelphia: "Philadelphia", chicago: "Chicago" };
const SCORERS = ["field_surprisal", "isolation_forest", "latent_distance", "recon_error"];

/** Horizontal per-field surprisal bars: one neutral hue (no city colour, never the status red),
 *  sorted high to low, value printed beside each bar. */
function SurprisalBars({ rec }: { rec: RecordResp }) {
  const fields = sortedFields(rec);
  const max = Math.max(...fields.map((f) => f.surprisal), 1e-9);
  return (
    <ol className="space-y-2" aria-label="Surprisal per field, highest first">
      {fields.map((f) => {
        const top = rec.reason_codes.includes(f.name);
        return (
          <li key={f.name} className="grid grid-cols-[9rem_1fr_3.5rem] items-center gap-3 text-sm">
            <span className={`truncate ${top ? "font-medium text-ink" : "text-ink-2"}`} title={f.name}>{f.name}</span>
            <span className="h-3 rounded-r-[3px] bg-surface-1" aria-hidden="true">
              <span className="block h-3 rounded-r-[3px]" style={{ width: `${(f.surprisal / max) * 100}%`, background: CONTEXT[0], minWidth: 2 }} />
            </span>
            <span className="text-right tabular text-ink-2">{formatSurprisal(f.surprisal)}</span>
          </li>
        );
      })}
    </ol>
  );
}

function RecordDetails({ city, row, note }: { city: string; row: number; note: string }) {
  const rec = useApi<RecordResp>(`/api/v1/record/${city}/${row}`);
  return (
    <Loader state={rec}>
      {(r) => (
        <div className="space-y-6">
          <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
            <InjectionLabel kind={r.kind} />
            <span className="text-sm text-ink-2">Group: <span className="text-ink">{groupShort(r.department)}</span></span>
          </div>
          <section aria-labelledby="rc">
            <h3 id="rc" className="text-base font-semibold">Top-3 reason codes</h3>
            <ol className="mt-2 list-inside list-decimal text-sm text-ink">{r.reason_codes.map((c) => <li key={c}>{c}</li>)}</ol>
          </section>
          <section aria-labelledby="sb">
            <h3 id="sb" className="text-base font-semibold">Surprisal per field</h3>
            <p className="mb-3 mt-1 text-xs text-ink-3">How unexpected each field is when the model hides it and predicts it from the others (−log p; squared error for amounts). Highest first.</p>
            <SurprisalBars rec={r} />
          </section>
          <section aria-labelledby="fv">
            <h3 id="fv" className="mb-2 text-base font-semibold">Field values</h3>
            <DataTable caption={`Field values of record ${r.row}`} minWidth={0}
              columns={[{ key: "f", label: "Field" }, { key: "v", label: "Value" }, { key: "s", label: "Surprisal", align: "right" }]}
              rows={r.fields.map((f) => ({
                f: f.name,
                v: <span className="break-words">{formatValue(f.value)}{typeof f.value === "number" && <span className="text-ink-3"> (scaled)</span>}</span>,
                s: formatSurprisal(f.surprisal),
              }))} />
            {r.fields.some((f) => typeof f.value === "number") && <p className="mt-2 text-xs text-ink-3">Note: {note}.</p>}
          </section>
        </div>
      )}
    </Loader>
  );
}

function AlertsView({ d, city }: { d: AlertsResp; city: string }) {
  const [open, setOpen] = useState<number | null>(null);
  const close = useCallback(() => setOpen(null), []);
  const found = d.alerts.filter((a) => a.kind !== "normal").length;
  const injected = d.n_injected.global + d.n_injected.local;
  return (
    <div className="space-y-8">
      <div className="grid gap-6 sm:grid-cols-3">
        <Figure label="Alerts (top 1% of E10)" value={String(d.n_alerts)} note={`of ${d.n_eval.toLocaleString("en-US")} records in E10`} />
        <Figure label="Injected anomalies among them" value={`${found} of ${injected}`}
          note={`${d.n_injected.global} global and ${d.n_injected.local} local were injected into E10`} />
        <Figure label="Model" value={`v1, seed ${d.seed}`} note={d.model_file} />
      </div>
      <Panel title={`Top ${d.n_alerts} records by ${SCORER_LABEL[d.scorer] ?? d.scorer}`}
        lead="Highest score first. Reason codes are the three fields with the highest surprisal (from the masked model) for every scorer, so they explain the record, not the scorer. Open a record to see all fields.">
        <DataTable caption={`Top 1% alerts on E10, ${CITY_NAME[city] ?? city}, ${SCORER_LABEL[d.scorer] ?? d.scorer}`}
          columns={[
            { key: "rank", label: "Rank", align: "right" }, { key: "record", label: "Record" },
            { key: "department", label: "Group" }, { key: "score", label: "Score", align: "right" },
            { key: "injected", label: "Injected?" }, { key: "reasons", label: "Top-3 reason codes" },
          ]}
          rows={d.alerts.map((a) => {
            const t = alertRowText(a);
            return {
              rank: t.rank,
              record: (
                <button type="button" onClick={() => setOpen(a.row)} aria-haspopup="dialog"
                  className="rounded-control px-1 text-beacon underline-offset-4 hover:underline">
                  <span className="sr-only">Open record </span>{t.record}
                </button>
              ),
              department: <span className={a.department ? "text-ink" : "text-ink-3"}>{t.department}</span>,
              score: t.score,
              injected: <InjectionLabel kind={a.kind} />,
              reasons: <span className="text-ink-2">{t.reasons}</span>,
            };
          })} />
        <div className="mt-3"><Source path={d.source} /></div>
      </Panel>
      <Drawer title={open !== null ? `Record ${open}` : ""} open={open !== null} onClose={close}>
        {open !== null && <RecordDetails city={city} row={open} note={d.numeric_note} />}
      </Drawer>
    </div>
  );
}

export default function Alerts() {
  const [city, setCity] = useState("philadelphia");
  const [scorer, setScorer] = useState("field_surprisal");
  const al = useApi<AlertsResp>(`/api/v1/alerts/${city}?scorer=${scorer}`);
  const available = al.data && !isPending(al.data) ? al.data.scorers : SCORERS;
  return (
    <div>
      <PageHeader title="Records flagged in E10"
        lead="What an auditor would review: the 1% of E10 records with the highest anomaly score. Whether a record was injected is the demo's ground truth; a real auditor would not see that column.">
        <div className="flex flex-wrap gap-3">
          <Segmented label="City" value={city} onChange={setCity}
            options={["philadelphia", "chicago"].map((c) => ({ value: c, label: CITY_NAME[c] }))} />
          <Segmented label="Scorer" value={scorer} onChange={setScorer}
            options={SCORERS.filter((s) => available.includes(s)).map((s) => ({ value: s, label: SCORER_LABEL[s] ?? s }))} />
        </div>
      </PageHeader>
      <Loader state={al} pendingTitle="Alerts not exported yet">{(d) => <AlertsView d={d} city={city} />}</Loader>
    </div>
  );
}
