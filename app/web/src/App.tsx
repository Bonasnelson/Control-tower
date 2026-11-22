import { useEffect, useState } from "react";
import Overview from "./pages/Overview";
import Stream from "./pages/Stream";
import Scoring from "./pages/Scoring";
import Alerts from "./pages/Alerts";
import Elliptic from "./pages/Elliptic";

export { PageHeader } from "./components/ui";

export type Page = "overview" | "stream" | "scoring" | "alerts" | "elliptic";
const NAV: { id: Page; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "stream", label: "Data stream" },
  { id: "scoring", label: "Scoring" },
  { id: "alerts", label: "Alerts" },
  { id: "elliptic", label: "Elliptic" },
];
const LATER = ["Drift gate", "Analyst feedback", "Canary and rollback"];

const fromHash = (): Page => {
  const h = window.location.hash.slice(1) as Page;
  return NAV.some((n) => n.id === h) ? h : "overview";
};

export default function App() {
  // The URL hash is the page state (deep links, back button, Ctrl+click all work).
  const [page, setPage] = useState<Page>(fromHash);
  useEffect(() => {
    const on = () => setPage(fromHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  useEffect(() => {
    document.title = `${NAV.find((n) => n.id === page)?.label} · Control Tower`;
    document.getElementById("main")?.focus({ preventScroll: true });
  }, [page]);
  const go = (p: Page) => { window.location.hash = p; };

  return (
    <div className="min-h-screen">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-control focus:bg-surface-2 focus:px-3 focus:py-2">
        Skip to content
      </a>
      <div className="mx-auto flex max-w-[1400px] flex-col lg:flex-row">
        <header className="border-b border-line px-4 py-4 lg:sticky lg:top-0 lg:h-screen lg:w-60 lg:shrink-0 lg:border-b-0 lg:border-r lg:px-6 lg:py-8">
          <div className="font-display text-lg font-semibold leading-tight text-ink">Control Tower</div>
          <div className="text-xs text-ink-3">Continual anomaly detection for city payments</div>
          <nav aria-label="Pages" className="mt-4 lg:mt-10">
            <ul className="flex flex-wrap gap-1 lg:flex-col">
              {NAV.map((n) => {
                const on = page === n.id;
                return (
                  <li key={n.id}>
                    <a href={`#${n.id}`} aria-current={on ? "page" : undefined}
                      className={`block whitespace-nowrap rounded-control border-l-2 px-3 py-2 text-sm transition-colors ${
                        on ? "border-beacon bg-surface-1 font-medium text-ink" : "border-transparent text-ink-2 hover:bg-surface-1 hover:text-ink"}`}>
                      {n.label}
                    </a>
                  </li>
                );
              })}
            </ul>
            <p className="mt-8 hidden px-3 text-xs text-ink-3 lg:block">Coming with milestones M3–M5</p>
            <ul className="hidden lg:mt-1 lg:block">
              {LATER.map((s) => (
                <li key={s} aria-disabled="true" className="cursor-not-allowed px-3 py-1.5 text-sm text-ink-3">{s}</li>
              ))}
            </ul>
          </nav>
          <p className="mt-10 hidden max-w-[24ch] text-xs text-ink-3 lg:block">
            COMP 5116 project. Every number here is read from a file in <code className="font-code">results/</code>.
          </p>
        </header>
        <main id="main" tabIndex={-1} className="min-w-0 flex-1 px-4 py-8 outline-none sm:px-8 lg:px-12 lg:py-10">
          {page === "overview" && <Overview go={go} />}
          {page === "stream" && <Stream />}
          {page === "scoring" && <Scoring />}
          {page === "alerts" && <Alerts />}
          {page === "elliptic" && <Elliptic />}
        </main>
      </div>
    </div>
  );
}
