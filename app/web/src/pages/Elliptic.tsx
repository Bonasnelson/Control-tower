import type { EllipticResp } from "../api";
import { useApi } from "../api";
import { DataTable, Loader, PageHeader, Panel, Source } from "../components/ui";

const STATUS_TEXT: Record<EllipticResp["status"], { title: string; body: string }> = {
  blocked: {
    title: "Blocked: the dataset is not in the repository",
    body: "The Elliptic files need a Kaggle login, so they cannot be downloaded automatically. Until they are in place this page shows no results, only what is ready.",
  },
  data_present: {
    title: "Data present, no results yet",
    body: "All three files are in place. No Elliptic experiment has been run yet, so there are no results to show.",
  },
  results_present: {
    title: "Data and result files present",
    body: "Result files exist (listed below). This page does not display them yet.",
  },
};

function FileStatus({ present }: { present: boolean }) {
  return present ? (
    <span className="inline-flex items-center gap-1.5 text-good-text">
      <svg aria-hidden="true" viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2"><path d="m3.5 8.5 3 3 6-7" strokeLinecap="round" strokeLinejoin="round" /></svg>
      Present
    </span>
  ) : (
    <span className="inline-flex items-center gap-1.5 text-ink-2">
      <svg aria-hidden="true" viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.6"><circle cx="8" cy="8" r="5.5" /><path d="M5.5 8h5" strokeLinecap="round" /></svg>
      Missing
    </span>
  );
}

function EllipticView({ e }: { e: EllipticResp }) {
  const st = STATUS_TEXT[e.status];
  const x = e.expected;
  return (
    <div className="space-y-8">
      <div role="status" className="rounded-panel border border-line-strong bg-surface-1 px-5 py-4">
        <h2 className="text-lg font-semibold">{st.title}</h2>
        <p className="mt-1 max-w-[72ch] text-sm text-ink-2">{st.body}</p>
      </div>

      <Panel title="Files needed" lead={<>Put these in <code className="font-code" translate="no">{e.data_dir}</code>.</>}>
        <DataTable caption="Elliptic files and whether they are present" minWidth={0}
          columns={[{ key: "f", label: "File" }, { key: "s", label: "Status" }]}
          rows={e.files.map((f) => ({ f: <code className="font-code" translate="no">{f.name}</code>, s: <FileStatus present={f.present} /> }))} />
        <ol className="mt-5 max-w-[72ch] list-inside list-decimal space-y-1 text-sm text-ink-2">
          <li>Sign in to Kaggle and open <a href={e.source.url} target="_blank" rel="noreferrer"
            className="text-beacon underline underline-offset-4">{e.source.name}</a>.</li>
          <li>Download the archive and unzip the three CSV files into <code className="font-code">{e.data_dir}</code>.</li>
          <li>Reload this page; the status changes to “Data present”. <code className="font-code">load_elliptic_cached</code> builds the cache <code className="font-code">data/processed/elliptic.npz</code> the first time it runs.</li>
        </ol>
      </Panel>

      <Panel title="What the data will look like"
        lead={<>Expected structure, taken from the loader (<code className="font-code">{x.from}</code>). These are format facts, not measurements.</>}>
        <dl className="grid max-w-3xl gap-x-8 gap-y-3 text-sm sm:grid-cols-[14rem_1fr]">
          <dt className="text-ink-2">Features per transaction</dt>
          <dd>{x.local_features ?? "?"} local + {x.aggregated_features ?? "?"} aggregated (expected)</dd>
          <dt className="text-ink-2">Time steps</dt><dd>{x.time_steps} (expected)</dd>
          <dt className="text-ink-2">Drift event</dt>
          <dd>Dark-market shutdown at time step {x.drift_step ?? "?"}, the real drift the project will evaluate on</dd>
          <dt className="text-ink-2">Labels</dt>
          <dd>illicit (class "1"), licit (class "2"), unlabelled ("unknown")</dd>
        </dl>
      </Panel>

      <Panel title="Ready and tested on synthetic data"
        lead="These functions in src/controltower/data/elliptic.py run on small synthetic CSVs written by the tests. They have not yet been run on the real files.">
        <ul className="space-y-3">
          {e.ready.map((r) => (
            <li key={r.function} className="text-sm">
              <code className="font-code text-ink" translate="no">{r.function}</code>
              <span className="text-ink-2"> — {r.what}</span>
            </li>
          ))}
        </ul>
        <p className="mt-5 text-sm text-ink-2">
          Tests in <code className="font-code">{e.tests.file}</code>, on {e.tests.data}:
        </p>
        <ul className="mt-2 flex flex-wrap gap-2">
          {e.tests.names.map((t) => (
            <li key={t}><code className="rounded-control bg-plane px-2 py-1 font-code text-xs text-ink-2" translate="no">{t}</code></li>
          ))}
        </ul>
        <p className="mt-4 text-xs text-ink-3">Run them with <code className="font-code">py -3.12 -m pytest -q tests/test_elliptic.py</code>.</p>
        {e.results_files.length > 0 && (
          <p className="mt-4 text-sm text-ink-2">Result files found: {e.results_files.map((f) => <Source key={f} path={`results/${f}`} />)}</p>
        )}
      </Panel>
    </div>
  );
}

export default function Elliptic() {
  const st = useApi<EllipticResp>("/api/elliptic");
  return (
    <div>
      <PageHeader title="Elliptic Bitcoin transactions"
        lead="A second, real-drift dataset: Bitcoin transactions grouped into time steps (49 expected), some labelled illicit or licit. It will test the same continual-learning ideas on a real market shutdown instead of a simulated department decay." />
      <Loader state={st}>{(e) => <EllipticView e={e} />}</Loader>
    </div>
  );
}
