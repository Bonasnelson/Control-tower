import { useState } from "react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { CitiesResp, ExperiencesResp } from "../api";
import { isPending, useApi } from "../api";
import { ChartPanel, DataTable, Figure, InjectionLabel, Loader, PageHeader, Runway, Segmented } from "../components/ui";
import { axisProps, tooltipProps } from "../components/ui";
import { CHART, CONTEXT, EMPHASIS } from "../palette";

const CITY_NAME: Record<string, string> = { philadelphia: "Philadelphia", chicago: "Chicago" };
const ANOM = ["GLOBAL_ANOM", "LOCAL_ANOM"];
const nf = new Intl.NumberFormat("en-US");

function Legend({ target }: { target: string }) {
  return (
    <ul className="mb-3 flex flex-wrap gap-x-6 gap-y-1 text-sm text-ink-2" aria-label="Legend">
      <li className="flex items-center gap-2"><span aria-hidden="true" className="h-3 w-3 rounded-[2px]" style={{ background: EMPHASIS }} />Target department {target}</li>
      <li className="flex items-center gap-2"><span aria-hidden="true" className="flex h-3 w-3 overflow-hidden rounded-[2px]">
        <span className="w-1/2" style={{ background: CONTEXT[0] }} /><span className="w-1/2" style={{ background: CONTEXT[1] }} /></span>Other departments</li>
    </ul>
  );
}

function StreamView({ d, city }: { d: ExperiencesResp; city: string }) {
  const target = d.target_dept ?? "";
  const others = d.departments.filter((x) => x !== target && !ANOM.includes(x));
  const order = [target, ...others]; // target at the baseline so its decay is readable
  const first = d.experiences[0], last = d.experiences[d.experiences.length - 1];
  const t1 = Number(first[target]), t10 = Number(last[target]);
  const cells = d.experiences.map((e) => ({ label: e.experience, count: Number(e[target]), share: Number(e[target]) / (t1 || 1) }));
  const name = CITY_NAME[city] ?? city;

  return (
    <div className="space-y-8">
      <Runway cells={cells} lit={cells.length - 1}
        caption={`${name}, target department ${target}: records per experience relative to E1.`} />
      <div className="grid gap-6 sm:grid-cols-3">
        <Figure label={`Target department ${target}, E1 → E10`} value={`${nf.format(t1)} → ${nf.format(t10)}`}
          note={`${((t10 / (t1 || 1)) * 100).toFixed(1)}% of its E1 count remains in E10`} />
        <Figure label="Records in E10" value={nf.format(Number(last.total))}
          note={`including ${nf.format(Number(last.GLOBAL_ANOM ?? 0))} global and ${nf.format(Number(last.LOCAL_ANOM ?? 0))} local injected anomalies`} />
        <Figure label="Departments" value={String(others.length + 1)} note={`${d.experiences.length} experiences, ${d.scenario} scenario`} />
      </div>

      <ChartPanel title="Records per department and experience"
        lead="Stacked counts. The target department sits at the bottom; the other departments are gray. Injected anomalies (E10 only) are listed in the table view."
        source={d.source}>
        <ChartPanel.Chart height={380}
          summary={`Stacked area chart of records per department for ${name}, E1 to E10. Target department ${target} falls from ${nf.format(t1)} to ${nf.format(t10)} records.`}>
          <Legend target={target} />
          <ResponsiveContainer width="100%" height="88%">
            <AreaChart data={d.experiences} margin={{ top: 4, right: 8, left: 0, bottom: 0 }} accessibilityLayer>
              <CartesianGrid stroke={CHART.grid} vertical={false} />
              <XAxis dataKey="experience" {...axisProps} />
              <YAxis {...axisProps} axisLine={false} width={56} tickFormatter={(v: number) => nf.format(v)} />
              <Tooltip {...tooltipProps} formatter={(v: number, k: string) => [nf.format(v), k === target ? `${k} (target)` : k]} />
              {order.map((k, i) => (
                <Area key={k} type="linear" dataKey={k} stackId="s" name={k} isAnimationActive={false}
                  fill={k === target ? EMPHASIS : CONTEXT[i % 2]} fillOpacity={1}
                  stroke={CHART.surface} strokeWidth={2} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </ChartPanel.Chart>
        <ChartPanel.Table>
          <DataTable caption={`Records per department and experience, ${name}`}
            columns={[{ key: "dept", label: "Department" },
              ...d.experiences.map((e) => ({ key: e.experience, label: e.experience, align: "right" as const }))]}
            rows={[
              ...order.map((k) => ({
                dept: k === target ? <span className="font-medium text-ink">{k} (target)</span> : <span className="text-ink-2">{k}</span>,
                ...Object.fromEntries(d.experiences.map((e) => [e.experience, nf.format(Number(e[k]))])),
              })),
              ...ANOM.filter((k) => d.departments.includes(k)).map((k) => ({
                dept: <InjectionLabel kind={k === "GLOBAL_ANOM" ? "global_anomaly" : "local_anomaly"} />,
                ...Object.fromEntries(d.experiences.map((e) => [e.experience, nf.format(Number(e[k]))])),
              })),
              { dept: <span className="font-medium">Total</span>,
                ...Object.fromEntries(d.experiences.map((e) => [e.experience, <span className="font-medium">{nf.format(Number(e.total))}</span>])) },
            ]} />
        </ChartPanel.Table>
      </ChartPanel>
    </div>
  );
}

export default function Stream() {
  const cities = useApi<CitiesResp>("/api/cities");
  // Philadelphia first (the paper's city), then any other configured city.
  const rank = (c: string) => { const i = ["philadelphia", "chicago"].indexOf(c); return i < 0 ? 99 : i; };
  const list = (cities.data && !isPending(cities.data) ? cities.data.cities.map((c) => c.city) : ["philadelphia", "chicago"])
    .slice().sort((a, b) => rank(a) - rank(b));
  const [city, setCity] = useState("philadelphia");
  const exp = useApi<ExperiencesResp>(`/api/experiences/${city}`);
  return (
    <div>
      <PageHeader title="The payment stream, E1 to E10"
        lead="Each experience is one time slice of the stream. In the linear scenario the target department keeps 100% of its slice in E1 and 1% in E10. The other departments are sampled at 90–100% of their slice where records exist; the table view shows the exact counts.">
        <Segmented label="City" value={city} onChange={setCity}
          options={list.map((c) => ({ value: c, label: CITY_NAME[c] ?? c }))} />
      </PageHeader>
      <Loader state={exp} pendingTitle="Experience table not built yet">{(d) => <StreamView d={d} city={city} />}</Loader>
    </div>
  );
}
