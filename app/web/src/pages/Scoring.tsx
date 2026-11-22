import { Bar, BarChart, CartesianGrid, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { MeanStd, ScoringResp } from "../api";
import { SCORER_LABEL, pm, useApi } from "../api";
import { ChartPanel, DataTable, Loader, PageHeader, Panel, Source, axisProps, tooltipProps } from "../components/ui";
import { CHART, CITY_COLOR } from "../palette";

const CITY_NAME: Record<string, string> = { philadelphia: "Philadelphia", chicago: "Chicago" };
const ORDER = ["philadelphia", "chicago"];
const METRIC_LABEL: Record<string, string> = { auprc: "AUPRC", recall_at_1pct: "Recall@1%" };

/** Category tick on two lines (first word, rest) so labels never collide on narrow screens. */
function WrappedTick({ x = 0, y = 0, payload }: { x?: number; y?: number; payload?: { value: string } }) {
  const words = String(payload?.value ?? "").split(" ");
  const lines = words.length > 1 ? [words[0], words.slice(1).join(" ")] : words;
  return (
    <text x={x} y={y + 12} textAnchor="middle" fill={CHART.ink3} fontSize={CHART.fontSize}>
      {lines.map((l, i) => <tspan key={i} x={x} dy={i === 0 ? 0 : 14}>{l}</tspan>)}
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
      <li className="flex items-center gap-2 text-ink-3">
        <svg aria-hidden="true" width="12" height="12" viewBox="0 0 12 12"><path d="M6 1v10M3 1h6M3 11h6" stroke={CHART.ink2} strokeWidth="1.5" /></svg>
        ± 1 std over seeds
      </li>
    </ul>
  );
}

function MetricChart({ s, cities, metric }: { s: ScoringResp; cities: string[]; metric: "auprc" | "recall_at_1pct" }) {
  const label = METRIC_LABEL[metric];
  const data = s.scorers.map((name) => ({
    scorer: SCORER_LABEL[name] ?? name,
    ...Object.fromEntries(cities.flatMap((c) => {
      const m: MeanStd | undefined = s.cities[c].scorers[name]?.[metric];
      return [[c, m?.mean ?? null], [`${c}_sd`, m?.std ?? 0]];
    })),
  }));
  const seeds = cities.map((c) => `${CITY_NAME[c] ?? c}: ${s.cities[c].seeds.length}`).join(", ");
  const best = cities.map((c) => {
    const top = [...s.scorers].sort((a, b) => (s.cities[c].scorers[b]?.[metric].mean ?? -1) - (s.cities[c].scorers[a]?.[metric].mean ?? -1))[0];
    return `${CITY_NAME[c] ?? c}: ${SCORER_LABEL[top]} ${pm(s.cities[c].scorers[top][metric], 2)}`;
  }).join("; ");
  return (
    <ChartPanel title={`${label} on E10 by scorer`}
      lead={`Bars are means over seeds (${seeds}); whiskers are ± 1 standard deviation. Highest mean: ${best}.`}
      source={cities.map((c) => s.cities[c].source).join(", ")}>
      <ChartPanel.Chart height={340}
        summary={`Grouped bar chart of ${label} per scorer for ${cities.map((c) => CITY_NAME[c] ?? c).join(" and ")}. ${best}.`}>
        <CityLegend cities={cities} />
        <ResponsiveContainer width="100%" height="86%">
          <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }} barGap={2} barCategoryGap="28%" accessibilityLayer>
            <CartesianGrid stroke={CHART.grid} vertical={false} />
            <XAxis dataKey="scorer" {...axisProps} interval={0} height={40} tick={<WrappedTick />} />
            <YAxis {...axisProps} axisLine={false} width={40} domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} />
            <Tooltip {...tooltipProps}
              formatter={(v: number, k: string, item: { payload?: Record<string, number> }) =>
                [`${v.toFixed(2)} ± ${(item.payload?.[`${k}_sd`] ?? 0).toFixed(2)}`, CITY_NAME[k] ?? k]} />
            {cities.map((c) => (
              <Bar key={c} dataKey={c} name={c} fill={CITY_COLOR[c]} radius={[4, 4, 0, 0]} isAnimationActive={false} maxBarSize={44}>
                <ErrorBar dataKey={`${c}_sd`} width={6} strokeWidth={1.5} stroke={CHART.ink2} direction="y" />
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer>
      </ChartPanel.Chart>
      <ChartPanel.Table>
        <DataTable caption={`${label} per scorer, mean ± std over seeds`}
          columns={[{ key: "scorer", label: "Scorer" },
            ...cities.map((c) => ({ key: c, label: CITY_NAME[c] ?? c, align: "right" as const }))]}
          rows={s.scorers.map((name) => ({
            scorer: SCORER_LABEL[name] ?? name,
            ...Object.fromEntries(cities.map((c) => [c, pm(s.cities[c].scorers[name]?.[metric], 2)])),
          }))} />
      </ChartPanel.Table>
    </ChartPanel>
  );
}

function FullTable({ s, city }: { s: ScoringResp; city: string }) {
  const c = s.cities[city];
  const train = c.scorers[s.scorers[0]]?.train_seconds;
  return (
    <Panel title={`All metrics, ${CITY_NAME[city] ?? city}`}
      lead={`Mean ± std over seeds ${c.seeds.join(", ")}; ${c.epochs ?? "?"} epochs. Training took ${train?.mean != null ? Math.round(train.mean) : "?"} s per seed on average; scoring times are for the whole of E10.`}>
      <DataTable caption={`Model v1 metrics on E10, ${CITY_NAME[city] ?? city}`}
        columns={[
          { key: "scorer", label: "Scorer" }, { key: "auprc", label: "AUPRC", align: "right" },
          { key: "r1", label: "Recall@1%", align: "right" }, { key: "dfp", label: "ΔFP", align: "right" },
          { key: "fp", label: "FP ratio", align: "right" }, { key: "fn", label: "FN ratio", align: "right" },
          { key: "t", label: "Score time (s)", align: "right" },
        ]}
        rows={s.scorers.map((name) => {
          const m = c.scorers[name];
          return {
            scorer: SCORER_LABEL[name] ?? name, auprc: pm(m?.auprc, 2), r1: pm(m?.recall_at_1pct, 2),
            dfp: pm(m?.delta_fp, 2), fp: pm(m?.fp_ratio, 2), fn: pm(m?.fn_ratio, 2), t: pm(m?.score_seconds, 1),
          };
        })} />
      <div className="mt-3"><Source path={c.source} /></div>
    </Panel>
  );
}

export default function Scoring() {
  const v1 = useApi<ScoringResp>("/api/v1/scoring");
  return (
    <div>
      <PageHeader title="Which score finds the anomalies?"
        lead="Model v1 (the masked-feature model, fine-tuned over E1–E10 with a random replay buffer) scored on E10 in four ways. Latent distance and isolation forest use E1–E9 as their reference set. Higher is better for AUPRC and Recall@1%." />
      <Loader state={v1} pendingTitle="Model v1 not trained yet">
        {(s) => {
          const cities = ORDER.filter((c) => c in s.cities).concat(Object.keys(s.cities).filter((c) => !ORDER.includes(c)));
          return (
            <div className="space-y-8">
              <MetricChart s={s} cities={cities} metric="auprc" />
              <MetricChart s={s} cities={cities} metric="recall_at_1pct" />
              <section aria-label="All metrics" className="space-y-8">
                {cities.map((c) => <FullTable key={c} s={s} city={c} />)}
              </section>
              <p className="max-w-[80ch] text-xs text-ink-3">
                ΔFP is the mean score of the shrinking department minus the highest mean score among the other
                departments; its scale depends on the scorer, so compare signs across scorers, not sizes. FP ratio: of
                the k highest-scored normal records (k = records of the shrinking department), the share from that
                department. FN ratio: leaving out global anomalies, of the k highest-scored records (k = local
                anomalies), the share that are not local anomalies.
              </p>
            </div>
          );
        }}
      </Loader>
    </div>
  );
}
