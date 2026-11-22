import type { Page } from "../App";
import type { BaselinesResp, ExperiencesResp, ScoringResp } from "../api";
import { SCORER_LABEL, fmt, isPending, pm, useApi } from "../api";
import { collapseSel } from "../baselines";
import { DataTable, Figure, Loader, PageHeader, Panel, PendingState, Runway, Source } from "../components/ui";

const CITY_NAME: Record<string, string> = { philadelphia: "Philadelphia", chicago: "Chicago" };


function PhiladelphiaRunway() {
  const exp = useApi<ExperiencesResp>("/api/experiences/philadelphia");
  const d = exp.data;
  if (!d || isPending(d) || !d.target_dept) return null;
  const t = d.target_dept;
  const first = Number(d.experiences[0][t]) || 1;
  const cells = d.experiences.map((e) => ({ label: e.experience, count: Number(e[t]), share: Number(e[t]) / first }));
  return (
    <Runway cells={cells} lit={cells.length - 1}
      caption={`Philadelphia, department ${t}: ${cells[0].count.toLocaleString()} records in E1, ${cells[cells.length - 1].count.toLocaleString()} in E10 (linear scenario). Models are evaluated on E10.`} />
  );
}

function BaselineSection() {
  const base = useApi<BaselinesResp>("/api/baselines");
  return (
    <Loader state={base} pendingTitle="Week 1 baselines not computed yet">
      {(b) => {
        const jel = b.rows.find((r) => r.regime === "JEL");
        const sel = b.rows.find((r) => r.regime === "SEL" && r.scenario === "linear");
        const rows = collapseSel(b.rows);
        return (
          <Panel title="Week 1: baseline reproduction"
            lead="One-hot autoencoder of Hemati et al. (2022) on Philadelphia, evaluated on E10. ΔFP above 0 means the shrinking department is ranked above every other department, which produces false positives.">
            <div className="grid gap-6 sm:grid-cols-3">
              <Figure label="ΔFP ×10³, trained on E10 only (SEL)" value={fmt(sel?.delta_fp_x1e3.mean, 3)}
                note={`Paper: ${b.paper.SEL.delta_fp_x1e3}`} />
              <Figure label="ΔFP ×10³, trained on E1–E10 (JEL)" value={fmt(jel?.delta_fp_x1e3.mean, 3)}
                note={`Paper: ${b.paper.JEL.delta_fp_x1e3}`} />
              <Figure label="AUPRC, JEL" value={fmt(jel?.auprc.mean, 3)} note={`± ${fmt(jel?.auprc.std, 3)} over ${jel?.seeds ?? "?"} seeds`} />
            </div>
            <div className="mt-8">
              <DataTable caption="Baseline reproduction, mean ± std over seeds"
                columns={[
                  { key: "regime", label: "Regime" }, { key: "scenario", label: "Scenario" },
                  { key: "seeds", label: "Seeds", align: "right" }, { key: "dfp", label: "ΔFP ×10³", align: "right" },
                  { key: "paper", label: "Paper ΔFP ×10³", align: "right" }, { key: "fp", label: "FP ratio", align: "right" },
                  { key: "fn", label: "FN ratio", align: "right" }, { key: "auprc", label: "AUPRC", align: "right" },
                  { key: "r1", label: "Recall@1%", align: "right" },
                ]}
                rows={rows.map((r) => ({
                  regime: <span className="font-medium">{r.regime}</span>, scenario: r.scenario, seeds: r.seeds,
                  dfp: pm(r.delta_fp_x1e3), paper: r.regime === "JEL" ? b.paper.JEL.delta_fp_x1e3 : b.paper.SEL.delta_fp_x1e3,
                  fp: pm(r.fp_ratio), fn: pm(r.fn_ratio), auprc: pm(r.auprc), r1: pm(r.recall_at_1pct),
                }))} />
            </div>
            <p className="mt-4 max-w-[80ch] text-xs text-ink-3">
              {rows.length < b.rows.length
                ? "SEL is shown once: it trains only on E10, which is the same in all three decay scenarios, so its results are identical. "
                : ""}
              Paper values: {b.paper.source}; the three SEL values are the paper's linear, exponential and instant runs.
            </p>
            <div className="mt-2"><Source path={b.source} /></div>
            {b.m1check && (
              <p className="mt-4 max-w-[80ch] text-sm text-ink-2">
                Sanity check of the masked-feature model (1 seed, {String(b.m1check.rows[0]?.epochs ?? "?")} epochs):{" "}
                {b.m1check.rows.map((r, i) => (
                  <span key={String(r.regime)}>{i > 0 && "; "}{String(r.regime)} AUPRC {fmt(r.auprc as number, 3)}, Recall@1% {fmt(r.recall_at_1pct as number, 2)}</span>
                ))}. <Source path={b.m1check.source} />
              </p>
            )}
          </Panel>
        );
      }}
    </Loader>
  );
}

function V1Section({ go }: { go: (p: Page) => void }) {
  const v1 = useApi<ScoringResp>("/api/v1/scoring");
  return (
    <Loader state={v1} pendingTitle="Model v1 not trained yet">
      {(s) => {
        const cities = ["philadelphia", "chicago"].filter((c) => c in s.cities).concat(Object.keys(s.cities).filter((c) => !["philadelphia", "chicago"].includes(c)));
        if (!cities.length) return <PendingState />;
        const c0 = s.cities[cities[0]];
        return (
          <Panel title="Week 2: model v1 scorer comparison"
            lead={`v1 is the masked-feature model fine-tuned over E1–E10 with a random replay buffer, scored on E10. Mean ± std over seeds ${c0.seeds.join(", ")}; ${c0.epochs ?? "?"} epochs.`}>
            <div>
              <DataTable caption="Model v1 scorers on E10, mean ± std over seeds"
                columns={[{ key: "scorer", label: "Scorer" }, ...cities.flatMap((c) => [
                  { key: `${c}_a`, label: `${CITY_NAME[c] ?? c} AUPRC`, align: "right" as const },
                  { key: `${c}_r`, label: `${CITY_NAME[c] ?? c} Recall@1%`, align: "right" as const }])]}
                rows={s.scorers.map((name) => ({
                  scorer: SCORER_LABEL[name] ?? name,
                  ...Object.fromEntries(cities.flatMap((c) => [
                    [`${c}_a`, pm(s.cities[c].scorers[name]?.auprc, 2)],
                    [`${c}_r`, pm(s.cities[c].scorers[name]?.recall_at_1pct, 2)]])),
                }))} />
            </div>
            <p className="mt-4 text-sm text-ink-2">
              <a href="#scoring" onClick={(e) => { e.preventDefault(); go("scoring"); }} className="text-beacon underline-offset-4 hover:underline">
                Open the full scorer comparison
              </a>{" "}with ΔFP, FP and FN ratios and run times.
            </p>
            <div className="mt-2 flex flex-wrap gap-x-4">{cities.map((c) => <Source key={c} path={s.cities[c].source} />)}</div>
          </Panel>
        );
      }}
    </Loader>
  );
}

export default function Overview({ go }: { go: (p: Page) => void }) {
  return (
    <div>
      <PageHeader title="Anomaly detection that keeps up with changing city payments"
        lead="Payments arrive in ten time slices. One department shrinks from slice to slice; a model that retrains as the data changes should stop flagging that department while still catching the injected anomalies.">
        <PhiladelphiaRunway />
      </PageHeader>
      <div className="space-y-8">
        <BaselineSection />
        <V1Section go={go} />
      </div>
    </div>
  );
}
