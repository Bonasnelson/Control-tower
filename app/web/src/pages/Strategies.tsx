import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ScoringResp, StrategiesResp } from "../api";
import { isPending, useApi } from "../api";
import { BETTER, METRIC_LABEL, STRATEGY_METRICS, type StrategyMetric, afterE10, fmt2, pct, strategyCells, usedForE10 } from "../strategies";
import { ChartPanel, DataTable, Loader, PageHeader, Panel, Segmented, Source, axisProps, tooltipProps } from "../components/ui";
import { CHART, CITY_COLOR, CONTEXT, EMPHASIS } from "../palette";

const CITY_NAME: Record<string, string> = { philadelphia: "Philadelphia", chicago: "Chicago" };
const ORDER = ["philadelphia", "chicago"];

/** True below 640 px: six two-line category labels do not fit, so charts switch to horizontal bars. */
function useNarrow() {
  const q = "(max-width: 639px)";
  const [narrow, setNarrow] = useState(() => window.matchMedia(q).matches);
  useEffect(() => {
    const m = window.matchMedia(q);
    const on = () => setNarrow(m.matches);
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, []);
  return narrow;
}

/** Two-line category tick: "Replay, random" -> Replay / random; "EWC + replay (random)" -> EWC + replay / (random). */
function TwoLineTick({ x = 0, y = 0, payload }: { x?: number; y?: number; payload?: { value: string } }) {
  const v = String(payload?.value ?? "");
  const i = v.search(/, | \(/);
  const lines = i < 0 ? [v] : [v.slice(0, i), v.slice(i).replace(/^, /, "")];
  return (
    <text x={x} y={y + 12} textAnchor="middle" fill={CHART.ink3} fontSize={CHART.fontSize}>
      {lines.map((l, k) => <tspan key={k} x={x} dy={k === 0 ? 0 : 14}>{l}</tspan>)}
    </text>
  );
}

function CityLegend({ cities }: { cities: string[] }) {
  return (
    <ul aria-label="Legend" className="mb-3 flex flex-wrap gap-x-6 gap-y-1 text-sm text-ink-2">
      {cities.map((c) => (
        <li key={c} className="flex items-center gap-2">
          <span aria-hidden="true" className="h-3 w-3 rounded-[2px]" style={{ background: CITY_COLOR[c] }} />{CITY_NAME[c] ?? c}
        </li>
      ))}
    </ul>
  );
}

function StrategyChart({ s, cities }: { s: StrategiesResp; cities: string[] }) {
  const [metric, setMetric] = useState<StrategyMetric>("auprc");
  const labels = s.cities[cities[0]].rows.map((r) => r.label);
  const data = labels.map((label) => ({
    label,
    ...Object.fromEntries(cities.map((c) => [c, s.cities[c].rows.find((r) => r.label === label)?.[metric] ?? null])),
  }));
  const better = BETTER[metric] === "higher" ? "Higher is better." : "Lower is better.";
  const narrow = useNarrow();
  const valueAxis = { ...axisProps, domain: (metric === "delta_fp" ? ["auto", "auto"] : [0, 1]) as [number | string, number | string],
    tickFormatter: (v: number) => v.toFixed(metric === "delta_fp" ? 0 : 2) };
  return (
    <ChartPanel title={`${METRIC_LABEL[metric]} on E10 by strategy`}
      lead={`One seed per bar (seed 0), so there are no error bars. ${better}`}
      source={cities.map((c) => s.cities[c].source).join(", ")}>
      <div className="mb-4">
        <Segmented label="Metric" value={metric} onChange={setMetric}
          options={STRATEGY_METRICS.map((m) => ({ value: m, label: METRIC_LABEL[m] }))} />
      </div>
      <ChartPanel.Chart height={narrow ? 520 : 360}
        summary={`Grouped bar chart of ${METRIC_LABEL[metric]} for six strategies in ${cities.map((c) => CITY_NAME[c] ?? c).join(" and ")}, seed 0.`}>
        <CityLegend cities={cities} />
        <ResponsiveContainer width="100%" height="86%">
          <BarChart data={data} layout={narrow ? "vertical" : "horizontal"} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
            barGap={2} barCategoryGap="24%" accessibilityLayer>
            <CartesianGrid stroke={CHART.grid} vertical={narrow} horizontal={!narrow} />
            {narrow ? (
              <>
                <XAxis type="number" {...valueAxis} />
                <YAxis type="category" dataKey="label" {...axisProps} width={112} interval={0} />
              </>
            ) : (
              <>
                <XAxis dataKey="label" {...axisProps} interval={0} height={40} tick={<TwoLineTick />} />
                <YAxis {...valueAxis} axisLine={false} width={44} />
              </>
            )}
            {metric === "delta_fp" && (narrow ? <ReferenceLine x={0} stroke={CHART.ink3} /> : <ReferenceLine y={0} stroke={CHART.ink3} />)}
            <Tooltip {...tooltipProps} formatter={(v: number, k: string) => [fmt2(v), CITY_NAME[k] ?? k]} />
            {cities.map((c) => (
              <Bar key={c} dataKey={c} name={c} fill={CITY_COLOR[c]} radius={narrow ? [0, 4, 4, 0] : [4, 4, 0, 0]} isAnimationActive={false} maxBarSize={40} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </ChartPanel.Chart>
      <ChartPanel.Table>
        <DataTable caption={`${METRIC_LABEL[metric]} by strategy, seed 0`}
          columns={[{ key: "label", label: "Strategy" }, ...cities.map((c) => ({ key: c, label: CITY_NAME[c] ?? c, align: "right" as const }))]}
          rows={labels.map((label) => ({
            label,
            ...Object.fromEntries(cities.map((c) => [c, fmt2(s.cities[c].rows.find((r) => r.label === label)?.[metric])])),
          }))} />
      </ChartPanel.Table>
    </ChartPanel>
  );
}

function FullTable({ s, city }: { s: StrategiesResp; city: string }) {
  const c = s.cities[city];
  return (
    <Panel title={`All metrics, ${CITY_NAME[city] ?? city}`}
      lead={`Seed ${c.seeds.join(", ")}; ${c.epochs} epochs per experience; scored with ${c.scorer.replace("_", " ")}.`}>
      <DataTable caption={`Strategy comparison on E10, ${CITY_NAME[city] ?? city}, seed 0`}
        columns={[
          { key: "label", label: "Strategy" }, { key: "auprc", label: "AUPRC", align: "right" },
          { key: "recall_at_1pct", label: "Recall@1%", align: "right" }, { key: "delta_fp", label: "ΔFP", align: "right" },
          { key: "fp_ratio", label: "FP ratio", align: "right" }, { key: "fn_ratio", label: "FN ratio", align: "right" },
          { key: "train_seconds", label: "Train time (s)", align: "right" },
        ]}
        rows={c.rows.map(strategyCells)} />
      <div className="mt-3"><Source path={c.source} /></div>
    </Panel>
  );
}

function BufferSection({ s, cities }: { s: StrategiesResp; cities: string[] }) {
  const [city, setCity] = useState(cities[0]);
  const b = s.cities[city]?.buffer;
  return (
    <section aria-labelledby="buf" className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 id="buf" className="text-lg font-semibold">What each replay rule keeps</h2>
          <p className="mt-1 max-w-[72ch] text-sm text-ink-2">
            Share of the buffer taken by the shrinking target department after each experience. The E9 buffer is the one
            replayed while the model trains on E10.
          </p>
        </div>
        <Segmented label="City for the buffer" value={city} onChange={setCity}
          options={cities.map((c) => ({ value: c, label: CITY_NAME[c] ?? c }))} />
      </div>
      {!b ? (
        <p className="text-sm text-ink-2">No buffer file for this city (<code className="font-code">results/dashboard/{city}_buffer.json</code>).</p>
      ) : (
        <BufferChart b={b} city={city} />
      )}
    </section>
  );
}

function BufferChart({ b, city }: { b: NonNullable<StrategiesResp["cities"][string]["buffer"]>; city: string }) {
  const narrow = useNarrow();
  const data = b.runs.map((r) => {
    const e9 = usedForE10(r.experiences)!;
    return { label: r.label, target: e9.target_share * 100, other: (e9.other_share + e9.injected_share) * 100, e9 };
  });
  const injectedE10 = b.runs.map((r) => ({ label: r.label, n: afterE10(r.experiences)?.injected ?? 0 })).filter((x) => x.n > 0);
  const exps = b.runs[0]?.experiences.map((e) => e.experience) ?? [];
  return (
    <ChartPanel title={`Buffer used while training E10 (state after E9), ${CITY_NAME[city] ?? city}`}
      lead={`Buffer size ${b.buffer_size ?? "?"}. Target department: ${b.target_department}. Seed 0.`} source={b.source}>
      <ChartPanel.Chart height={300}
        summary={`Stacked bars of the E9 replay buffer per rule: share of target department ${b.target_department} versus other departments.`}>
        <ul aria-label="Legend" className="mb-3 flex flex-wrap gap-x-6 gap-y-1 text-sm text-ink-2">
          <li className="flex items-center gap-2"><span aria-hidden="true" className="h-3 w-3 rounded-[2px]" style={{ background: EMPHASIS }} />Target department {b.target_department}</li>
          <li className="flex items-center gap-2"><span aria-hidden="true" className="h-3 w-3 rounded-[2px]" style={{ background: CONTEXT[0] }} />Other departments</li>
        </ul>
        <ResponsiveContainer width="100%" height="84%">
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 0 }} accessibilityLayer>
            <CartesianGrid stroke={CHART.grid} horizontal={false} />
            <XAxis type="number" domain={[0, 100]} {...axisProps} tickFormatter={(v: number) => `${v}%`} />
            <YAxis type="category" dataKey="label" {...axisProps} width={narrow ? 104 : 150} />
            <Tooltip {...tooltipProps} formatter={(v: number, k: string) => [`${v.toFixed(1)}%`, k === "target" ? `Target ${b.target_department}` : "Other departments"]} />
            <Bar dataKey="target" stackId="b" fill={EMPHASIS} isAnimationActive={false} stroke={CHART.surface} strokeWidth={2} />
            <Bar dataKey="other" stackId="b" fill={CONTEXT[0]} isAnimationActive={false} stroke={CHART.surface} strokeWidth={2} />
          </BarChart>
        </ResponsiveContainer>
      </ChartPanel.Chart>
      <ChartPanel.Table>
        <DataTable caption={`Target-department share of the replay buffer after each experience, ${CITY_NAME[city] ?? city}`}
          columns={[{ key: "label", label: "Rule" }, ...exps.map((e) => ({ key: e, label: e, align: "right" as const })),
            { key: "inj", label: "Injected in E10 buffer", align: "right" }]}
          rows={b.runs.map((r) => ({
            label: r.label,
            ...Object.fromEntries(r.experiences.map((e) => [e.experience, pct(e.target_share)])),
            inj: String(afterE10(r.experiences)?.injected ?? 0),
          }))} />
      </ChartPanel.Table>
      <p className="mt-4 max-w-[80ch] text-xs text-ink-3">
        The E10 entry is the buffer after its refresh at the end of E10; no model shown here trained on it.
        {injectedE10.length > 0
          ? ` It contains injected anomalies for ${injectedE10.map((x) => `${x.label} (${x.n})`).join(", ")}, because those rules select from E10 records.`
          : " It contains no injected anomalies."}
      </p>
    </ChartPanel>
  );
}

export default function Strategies() {
  const st = useApi<StrategiesResp>("/api/strategies");
  const v1 = useApi<ScoringResp>("/api/v1/scoring");
  const v1s = v1.data && !isPending(v1.data) ? v1.data : null;
  return (
    <div>
      <PageHeader title="Which continual-learning strategy helps?"
        lead="The masked-feature model fine-tuned over E1–E10 with each strategy, then scored on E10 with field surprisal. Replay keeps 2,000 earlier records and mixes them into training; EWC penalises moving the weights that mattered for earlier experiences." />
      <Loader state={st} pendingTitle="Strategy runs not done yet">
        {(s) => {
          const cities = ORDER.filter((c) => c in s.cities).concat(Object.keys(s.cities).filter((c) => !ORDER.includes(c)));
          return (
            <div className="space-y-8">
              <div role="note" className="rounded-panel border border-line-strong bg-surface-1 px-5 py-4">
                <h2 className="text-lg font-semibold">One seed (seed 0), no variance</h2>
                <p className="mt-1 max-w-[76ch] text-sm text-ink-2">
                  Each number comes from a single run, so small differences between strategies may be run-to-run noise.
                  {v1s && <> For scale: over seeds {v1s.cities[cities[0]]?.seeds.join(", ")}, model v1 (the same setup as “Replay, random”) had a field-surprisal AUPRC standard deviation of{" "}
                    {cities.map((c, i) => <span key={c}>{i > 0 && " and "}{fmt2(v1s.cities[c]?.scorers.field_surprisal?.auprc.std)} ({CITY_NAME[c] ?? c})</span>)}.</>}
                </p>
              </div>
              <StrategyChart s={s} cities={cities} />
              {cities.map((c) => <FullTable key={c} s={s} city={c} />)}
              <BufferSection s={s} cities={cities} />
            </div>
          );
        }}
      </Loader>
    </div>
  );
}
