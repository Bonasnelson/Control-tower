// Typed client for the read-only FastAPI backend (app/api/main.py).
import { useEffect, useState } from "react";

export type Pending = { status: "pending"; missing: string; command: string };
export type MeanStd = { mean: number | null; std: number | null; n?: number };

export type CitiesResp = {
  cities: { city: string; n_departments: number; target_dept: string; categorical: string[]; numerical: string[] }[];
  scenarios: { name: string; reduction_type: string; n_experiences: number; target_dept_data_perc?: number[] }[];
};

export type ExperiencesResp = {
  status: "ok"; city: string; scenario: string; source: string; departments: string[]; target_dept: string | null;
  experiences: ({ experience: string; total: number } & Record<string, number | string>)[];
};

export type BaselineRow = {
  regime: string; scenario: string; seeds: number;
  delta_fp_x1e3: MeanStd; fp_ratio: MeanStd; fn_ratio: MeanStd; auprc: MeanStd; recall_at_1pct: MeanStd;
};
export type BaselinesResp = {
  status: "ok"; source: string; rows: BaselineRow[];
  /** null when results/paper_reference_philadelphia.json is missing */
  paper: { source: string; file: string; verified: boolean; verified_by: string; verified_date: string;
    verified_note: string; unit_note: string;
    SEL: { delta_fp_x1e3: string; L_rec_target_x1e3: number }; JEL: { delta_fp_x1e3: string; L_rec_target_x1e3: number } } | null;
  m1check: { source: string; rows: Record<string, number | string | null>[] } | null;
};

export type ScorerStats = Record<string, MeanStd>;
export type ScoringResp = {
  status: "ok"; scorers: string[];
  cities: Record<string, { source: string; seeds: number[]; epochs: number | null; scorers: Record<string, ScorerStats> }>;
};

export type Alert = { rank: number; row: number; score: number; kind: string; department: string | null; reason_codes: string[] };
export type AlertsResp = {
  status: "ok"; city: string; seed: number; scorer: string; scorers: string[]; source: string; model_file: string;
  experience: string; n_eval: number; budget: number; n_alerts: number; n_injected: { global: number; local: number };
  target_department: string; numeric_note: string; alerts: Alert[];
};
export type RecordResp = {
  status: "ok"; city: string; row: number; kind: string; department: string | null; reason_codes: string[];
  fields: { name: string; value: string | number; surprisal: number }[];
};

export type Loadable<T> = { data?: T | Pending; error?: string; loading: boolean };

export function isPending(x: unknown): x is Pending {
  return typeof x === "object" && x !== null && (x as { status?: string }).status === "pending";
}

export function useApi<T>(path: string | null): Loadable<T> {
  const [state, setState] = useState<Loadable<T>>({ loading: !!path });
  useEffect(() => {
    if (!path) return;
    let alive = true;
    setState({ loading: true });
    fetch(path)
      .then(async (r) => {
        if (!r.ok) throw new Error(`${r.status} ${(await r.json().catch(() => ({}))).detail ?? r.statusText}`);
        return r.json();
      })
      .then((d) => alive && setState({ data: d, loading: false }))
      .catch((e: Error) => alive && setState({ error: e.message, loading: false }));
    return () => { alive = false; };
  }, [path]);
  return state;
}

export const SCORER_LABEL: Record<string, string> = {
  recon_error: "Reconstruction error",
  field_surprisal: "Field surprisal",
  latent_distance: "Latent k-NN distance",
  isolation_forest: "Isolation forest",
};

export const fmt = (v: number | null | undefined, d = 3) => (v === null || v === undefined ? "—" : v.toFixed(d));
export const pm = (m?: MeanStd, d = 3) => (m && m.mean !== null ? `${fmt(m.mean, d)} ± ${fmt(m.std, d)}` : "—");


export type EllipticResp = {
  status: "blocked" | "data_present" | "results_present";
  data_dir: string;
  files: { name: string; present: boolean }[];
  source: { name: string; url: string; note: string };
  expected: { local_features: number | null; aggregated_features: number | null; time_steps: string;
    drift_step: number | null; labels: Record<string, number>; from: string };
  ready: { function: string; what: string }[];
  tests: { file: string; names: string[]; data: string };
  results_files: string[];
};

export type StrategyRow = {
  strategy: string; buffer_rule: string | null; label: string; seeds: number[];
  auprc: MeanStd; recall_at_1pct: MeanStd; delta_fp: MeanStd; fp_ratio: MeanStd; fn_ratio: MeanStd; train_seconds: MeanStd;
};
export type BufferExp = { experience: string; size: number; target: MeanStd; injected: MeanStd; target_share: MeanStd };
export type StrategiesResp = {
  status: "ok"; metrics: string[];
  cities: Record<string, {
    source: string; seeds: number[]; epochs: number; scorer: string; rows: StrategyRow[];
    buffer: { source: string; target_department: string; buffer_size: number | null;
      runs: { strategy: string; buffer_rule: string; seeds: number[]; label: string; experiences: BufferExp[] }[] } | null;
  }>;
};

export type SpreadRow = {
  strategy: string; buffer_rule: string | null; label: string; seeds: number[];
  auprc: MeanStd; recall_at_1pct: MeanStd; delta_fp: MeanStd; fn_ratio: MeanStd;
  buffer_injected_used_total: MeanStd | null; buffer_injected_after_E9: MeanStd | null; filter_dropped_injected_total: MeanStd | null;
};
export type SpreadResp = {
  status: "ok"; scenario: string; description: string;
  cities: Record<string, { source: string; scenario: string; seeds: number[]; rows: SpreadRow[] }>;
};
