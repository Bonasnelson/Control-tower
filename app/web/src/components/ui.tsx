// Shared UI components (T2). Tokens from tailwind.config.js; rules from the installed
// frontend-design, web-design-guidelines and composition-patterns skills (React 18: no
// React-19-only APIs) and the dataviz skill (charts always paired with a table view).
import { createContext, useContext, useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { createPortal } from "react-dom";
import type { Loadable, Pending } from "../api";
import { isPending } from "../api";
import { CHART } from "../palette";

/* ------------------------------------------------------------------ layout */

export function PageHeader({ title, lead, children }: { title: string; lead?: ReactNode; children?: ReactNode }) {
  return (
    <header className="mb-8">
      <h1 className="text-xl font-semibold tracking-tight sm:text-[2rem] sm:leading-[2.4rem]">{title}</h1>
      {lead && <p className="mt-2 max-w-[68ch] text-base text-ink-2">{lead}</p>}
      {children && <div className="mt-5">{children}</div>}
    </header>
  );
}

/** Flat panel. Title is an h2; ``aside`` holds controls or the source path. */
export function Panel({ title, lead, aside, children, className = "" }: {
  title?: string; lead?: ReactNode; aside?: ReactNode; children: ReactNode; className?: string;
}) {
  const id = useId();
  return (
    <section aria-labelledby={title ? id : undefined}
      className={`rounded-panel border border-line bg-surface-1 ${className}`}>
      {(title || aside) && (
        <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-2 border-b border-line px-5 py-4">
          <div className="min-w-0">
            {title && <h2 id={id} className="text-lg font-semibold">{title}</h2>}
            {lead && <p className="mt-1 max-w-[72ch] text-sm text-ink-2">{lead}</p>}
          </div>
          {aside && <div className="flex flex-wrap items-center gap-3">{aside}</div>}
        </div>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}
/** Backwards-compatible name used by the first scaffold. */
export const Card = ({ title, subtitle, right, children, className }: {
  title?: string; subtitle?: ReactNode; right?: ReactNode; children: ReactNode; className?: string;
}) => <Panel title={title} lead={subtitle} aside={right} className={className}>{children}</Panel>;

/** A measured headline figure with its label and a qualifier (seeds, source...). */
export function Figure({ label, value, note }: { label: string; value: ReactNode; note?: ReactNode }) {
  return (
    <div className="min-w-0 border-l-2 border-line-strong pl-4">
      <div className="text-sm text-ink-2">{label}</div>
      <div className="mt-1 font-display text-figure font-semibold text-ink">{value}</div>
      {note && <div className="mt-1 text-xs text-ink-3">{note}</div>}
    </div>
  );
}
/** Old name kept so existing pages compile until T3/T4 revise them. */
export const Stat = ({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode; tone?: string }) =>
  <Figure label={label} value={value} note={hint} />;

export function Source({ path }: { path: string }) {
  return <span className="text-xs text-ink-3">Source: <code className="font-code" translate="no">{path}</code></span>;
}

/* ---------------------------------------------------------- data states */

export function PendingState({ p, title = "Not computed yet" }: { p?: Pending; title?: string }) {
  return (
    <div role="status" className="rounded-panel border border-dashed border-line-strong px-6 py-10">
      <h3 className="text-lg font-semibold">{title}</h3>
      <p className="mt-2 max-w-[64ch] text-sm text-ink-2">
        <code className="font-code" translate="no">{p?.missing ?? "results"}</code> does not exist yet, so nothing is
        shown here. Every figure on this dashboard is read from a file the pipeline wrote.
      </p>
      <p className="mt-4 text-sm text-ink-2">Create it with:</p>
      <pre className="mt-2 overflow-x-auto rounded-control bg-plane px-3 py-2 font-code text-sm text-ink"
        translate="no">{p?.command ?? "py -3.12 scripts/run_v1.py"}</pre>
    </div>
  );
}

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div role="alert" className="flex gap-3 rounded-panel border border-critical/60 bg-critical/10 px-4 py-3 text-sm">
      <StatusIcon kind="critical" />
      <div>
        <p className="font-medium text-ink">The API did not answer: {message}</p>
        <p className="mt-1 text-ink-2">Start the backend with <code className="font-code">py -3.12 scripts/run_dashboard.py</code> and reload.</p>
      </div>
    </div>
  );
}

/** Renders loading / error / pending states; ``children`` receives the loaded data. */
export function Loader<T>({ state, children, pendingTitle }: {
  state: Loadable<T>; children: (d: T) => ReactNode; pendingTitle?: string;
}) {
  if (state.loading)
    return <div aria-busy="true" className="h-40 rounded-panel bg-surface-1"><span className="sr-only">Loading…</span></div>;
  if (state.error) return <ErrorBanner message={state.error} />;
  if (isPending(state.data)) return <PendingState p={state.data} title={pendingTitle} />;
  return <>{state.data ? children(state.data as T) : null}</>;
}

/* ------------------------------------------------------------- status */

function StatusIcon({ kind }: { kind: "critical" | "local" | "none" }) {
  if (kind === "none")
    return <svg aria-hidden="true" viewBox="0 0 16 16" className="h-4 w-4 shrink-0 text-ink-3"><circle cx="8" cy="8" r="3" fill="currentColor" /></svg>;
  if (kind === "local")
    return (
      <svg aria-hidden="true" viewBox="0 0 16 16" className="h-4 w-4 shrink-0 text-critical-text" fill="none" stroke="currentColor" strokeWidth="1.6">
        <path d="M8 2.5 14 13H2L8 2.5Z" strokeLinejoin="round" />
      </svg>
    );
  return (
    <svg aria-hidden="true" viewBox="0 0 16 16" className="h-4 w-4 shrink-0 text-critical-text">
      <path d="M8 1.5 14.5 13.5h-13L8 1.5Z" fill="currentColor" />
      <path d="M8 6v3.5M8 11.3v.2" stroke="#15233a" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

/** Ground truth of the demo: was the record injected? Icon + label, never colour alone.
 *  Uses the reserved critical status colour, so it never appears inside a chart that
 *  also uses series-2 (Chicago); see docs/dashboard_plan.md §5.1. */
export function InjectionLabel({ kind }: { kind: string }) {
  const map: Record<string, [string, "critical" | "local" | "none", string]> = {
    global_anomaly: ["Injected global anomaly", "critical", "text-critical-text"],
    local_anomaly: ["Injected local anomaly", "local", "text-critical-text"],
    normal: ["Not injected", "none", "text-ink-3"],
  };
  const [label, icon, cls] = map[kind] ?? [kind, "none", "text-ink-2"];
  return <span className={`inline-flex items-center gap-1.5 whitespace-nowrap text-sm ${cls}`}><StatusIcon kind={icon} />{label}</span>;
}
/** Old name kept for the first scaffold. */
export const Badge = InjectionLabel;

/* ----------------------------------------------------------- controls */

/** Single-choice control (WAI-ARIA radio group): arrow keys move, Tab leaves. */
export function Segmented<T extends string>({ label, value, options, onChange }: {
  label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const onKey = (e: KeyboardEvent, i: number) => {
    const d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    const j = (i + d + options.length) % options.length;
    onChange(options[j].value);
    refs.current[j]?.focus();
  };
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex max-w-full flex-wrap rounded-control border border-line bg-plane p-0.5">
      {options.map((o, i) => {
        const on = o.value === value;
        return (
          <button key={o.value} ref={(el) => { refs.current[i] = el; }} type="button" role="radio" aria-checked={on}
            tabIndex={on ? 0 : -1} onClick={() => onChange(o.value)} onKeyDown={(e) => onKey(e, i)}
            className={`touch-manipulation rounded-[5px] px-3 py-1.5 text-sm transition-colors ${
              on ? "bg-surface-2 font-medium text-ink shadow-[inset_0_-2px_0_var(--beacon)]" : "text-ink-2 hover:text-ink"}`}>
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

/* --------------------------------------------- chart + table (compound) */

type ViewCtx = { view: "chart" | "table"; setView: (v: "chart" | "table") => void };
const ChartViewContext = createContext<ViewCtx | null>(null);
const useView = () => {
  const c = useContext(ChartViewContext);
  if (!c) throw new Error("ChartPanel.* must be used inside <ChartPanel>");
  return c;
};

function ChartPanelRoot({ title, lead, source, children }: { title: string; lead?: ReactNode; source?: string; children: ReactNode }) {
  const [view, setView] = useState<"chart" | "table">("chart");
  return (
    <ChartViewContext.Provider value={{ view, setView }}>
      <Panel title={title} lead={lead}
        aside={<Segmented label={`${title}: view`} value={view} onChange={setView}
          options={[{ value: "chart", label: "Chart" }, { value: "table", label: "Table" }]} />}>
        {children}
        {source && <div className="mt-4"><Source path={source} /></div>}
      </Panel>
    </ChartViewContext.Provider>
  );
}
function ChartSlot({ children, height = 360, summary }: { children: ReactNode; height?: number; summary: string }) {
  const { view } = useView();
  if (view !== "chart") return null;
  return <figure role="img" aria-label={summary} style={{ height }} className="w-full">{children}</figure>;
}
function TableSlot({ children }: { children: ReactNode }) {
  const { view } = useView();
  return view === "table" ? <>{children}</> : null;
}
/** Every chart ships with a table of the same numbers (dataviz relief rule). */
export const ChartPanel = Object.assign(ChartPanelRoot, { Chart: ChartSlot, Table: TableSlot });

/* --------------------------------------------------------------- tables */

export function DataTable({ caption, columns, rows, minWidth = 560 }: {
  caption: string;
  /** px below which the table scrolls inside its region (0 = always fit the container). */
  minWidth?: number;
  columns: { key: string; label: string; align?: "left" | "right" }[];
  rows: Record<string, ReactNode>[];
}) {
  // Wide tables scroll inside a focusable, labelled region instead of clipping (keyboard users can scroll it).
  return (
    <div role="region" tabIndex={0} aria-label={caption} className="overflow-x-auto rounded-control">
    <table className="w-full border-collapse text-sm" style={{ minWidth }}>
      <caption className="sr-only">{caption}</caption>
      <thead>
        <tr>
          {columns.map((c) => (
            <th key={c.key} scope="col"
              className={`border-b border-line-strong px-3 py-2 font-medium text-ink-2 ${c.align === "right" ? "text-right" : "text-left"}`}>
              {c.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i} className="border-b border-line last:border-b-0">
            {columns.map((c) => (
              <td key={c.key} className={`px-3 py-2.5 ${c.align === "right" ? "whitespace-nowrap text-right" : "text-left"}`}>{r[c.key]}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
    </div>
  );
}

/* --------------------------------------------------------------- drawer */

/** Modal side drawer: focus moves in, Tab is trapped, Esc closes, focus returns. */
export function Drawer({ title, open, onClose, children }: { title: string; open: boolean; onClose: () => void; children: ReactNode }) {
  const panel = useRef<HTMLDivElement>(null);
  const id = useId();
  useEffect(() => {
    if (!open) return;
    const before = document.activeElement as HTMLElement | null;
    panel.current?.focus();
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); onClose(); return; }
      if (e.key !== "Tab" || !panel.current) return;
      const f = panel.current.querySelectorAll<HTMLElement>('button, [href], [tabindex]:not([tabindex="-1"])');
      if (!f.length) return;
      const first = f[0], last = f[f.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    };
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("keydown", onKey); before?.focus(); };
  }, [open, onClose]);
  if (!open) return null;
  // Portal to <body>: escapes parent layout (e.g. space-y margins) and stacking contexts.
  return createPortal(
    <div className="fixed inset-0 z-40 flex justify-end">
      <div aria-hidden="true" className="absolute inset-0 bg-plane/70" onClick={onClose} />
      <div ref={panel} role="dialog" aria-modal="true" aria-labelledby={id} tabIndex={-1}
        className="relative flex h-full w-full max-w-xl flex-col overflow-y-auto overscroll-contain border-l border-line-strong bg-surface-2 outline-none">
        <div className="flex items-start justify-between gap-4 border-b border-line px-6 py-5">
          <h2 id={id} className="text-lg font-semibold">{title}</h2>
          <button type="button" onClick={onClose} aria-label="Close record details"
            className="rounded-control px-2 py-1 text-ink-2 hover:bg-surface-1 hover:text-ink">
            <svg aria-hidden="true" viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M3.5 3.5l9 9m0-9-9 9" strokeLinecap="round" /></svg>
          </button>
        </div>
        <div className="px-6 py-5">{children}</div>
      </div>
    </div>,
    document.body,
  );
}

/* --------------------------------------------- experience runway (signature) */

/** The one bold element: E1→E10 as runway cells, height = a measured share (0..1),
 *  the evaluated experience lit in primary ink (beacon is reserved for interaction). Values come from results/experiences_<city>_linear.csv. */
export function Runway({ cells, lit, caption }: {
  cells: { label: string; share: number; count: number }[]; lit: number; caption: string;
}) {
  return (
    <figure className="max-w-3xl">
      <ol className="grid grid-cols-10 items-end gap-1.5" aria-label={caption}>
        {cells.map((c, i) => {
          const on = i === lit;
          return (
            <li key={c.label} className="flex flex-col items-center gap-1.5">
              <div className="relative flex h-14 w-full items-end overflow-hidden rounded-[3px] bg-surface-1">
                <div className={on ? "w-full bg-ink" : "w-full bg-series-1"} style={{ height: `${Math.max(4, c.share * 100)}%` }} />
              </div>
              <span className={`text-xs tabular ${on ? "font-semibold text-ink" : "text-ink-3"}`}>{c.label}</span>
              <span className="sr-only">{`${c.count.toLocaleString()} records (${Math.round(c.share * 100)}% of the E1 slice)`}</span>
            </li>
          );
        })}
      </ol>
      <figcaption className="mt-2 text-xs text-ink-3">{caption}</figcaption>
    </figure>
  );
}

/* -------------------------------------------------------- Recharts chrome */

export const tooltipProps = {
  contentStyle: { background: CHART.surface2, border: `1px solid ${CHART.axis}`, borderRadius: 6, fontSize: 12, color: CHART.ink },
  labelStyle: { color: CHART.ink, fontWeight: 600 },
  itemStyle: { color: CHART.ink2, padding: 0 },
  cursor: { stroke: CHART.axis, fill: "rgba(255,255,255,0.04)" },
} as const;
export const axisProps = { stroke: CHART.axis, tick: { fill: CHART.ink3, fontSize: CHART.fontSize }, tickLine: false } as const;
