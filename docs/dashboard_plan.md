# Demo dashboard plan (draft for approval)

Status: plan only. Nothing below is built beyond the partial scaffold listed in "Current state". The rule throughout
is that the dashboard shows only numbers read from files the pipeline wrote; anything missing renders an empty state
naming the file and the command that produces it.

## 1. Goal and users

| User | Situation | What they must be able to see |
|---|---|---|
| Course instructor / TA | Watches the final demo and grades it, may open it afterwards | That the Week-1 baselines reproduce the paper (SEL vs JEL, ΔFP, with paper values beside ours); what v1 is and how its scorers compare; that every number has a source file |
| The two students (presenters) | Walk through the story live in ~5 min | One page per step of the story: data stream → baselines → v1 scorers → strategies → alerts with reasons; fast city switching |
| Auditor / analyst (imagined end user) | Reviews the records flagged this period | A ranked alert list for the latest experience, why each record was flagged (per-field surprisal, top-3 reason codes), whether the flag is an injected anomaly (demo ground truth) |

Content that must appear:
1. Philadelphia and Chicago data: departments, the decaying target department, records per experience E1–E10.
2. SEL vs JEL baseline reproduction vs Hemati et al. (2022) Table 2.
3. Model v1 scorer comparison (recon_error, field_surprisal, latent_distance, isolation_forest): AUPRC, Recall@1%, ΔFP, FP/FN ratio, mean ± std over seeds, both cities.
4. Continual strategies: fine-tuning with none / EWC / replay (random, diversity, high_error) / EWC+replay.
5. Anomaly scores and reason codes on E10 (alerts + record detail).
6. Elliptic: shown as "blocked: needs the Kaggle files"; the pipeline is ready and tested on synthetic data only.
7. Placeholders for M3–M5 (drift gate, analyst feedback, canary/rollback), disabled.

## 2. Pages and components

| Page | Components | Data |
|---|---|---|
| Overview | Project one-liner; 4 stat tiles (SEL ΔFP, JEL ΔFP, JEL AUPRC, v1 status); baseline table ours vs paper; M1 sanity-check card; "what is v1" note | `/api/baselines`, `/api/v1/scoring` |
| Data stream | City switch; stacked area chart (records per department E1→E10, target department highlighted, anomalies in E10); data table fallback of the same numbers; city facts (records, departments, fields) | `/api/cities`, `/api/experiences/{city}` |
| Scoring (v1) | City-grouped bar charts for AUPRC and Recall@1% with ± std error bars; full metrics table (all 5 metrics + score time); note on reference set (E1..E9) | `/api/v1/scoring` |
| Strategies | Table + bar chart: none / EWC / replay×3 rules / EWC+replay for AUPRC, Recall@1%, ΔFP; buffer composition per rule (share of each department in the buffer after E9) | `/api/strategies` (new; depends on new runs) |
| Alerts | City + scorer switch; top-1% alert table (rank, row, department, score, injected flag, reason codes); record drawer with per-field surprisal bars and decoded field values | `/api/v1/alerts/{city}`, `/api/v1/record/{city}/{row}` |
| Elliptic | Blocked state with the expected files and the pipeline that is ready (`load_elliptic_cached`, `edge_features`, `delayed_label_stream`) | `/api/elliptic` (returns pending until the data exist) |
| M3–M5 (nav only) | Disabled entries "coming with M3–M5" | none |

Shared components: page header, card, stat tile, segmented control, empty/pending state (file + command), error
banner, source footnote (file path on every card), chart + table pair.

## 3. Data contract

| Artifact | Schema (key fields) | Producer | Exists today? |
|---|---|---|---|
| `configs/<city>.yaml`, `configs/scenarios/*.yaml` | city, n_departments, target_dept, categorical, numerical; reduction_type, n_experiences, target_dept_data_perc | hand-written | yes |
| `results/experiences_<city>_linear.csv` | rows E1..E10; one column per department + GLOBAL_ANOM, LOCAL_ANOM, total | `scripts/prepare_data.py` | yes (both cities) |
| `results/summary_philadelphia_baseline_ae.md` (+ `baselines_philadelphia_baseline_ae_{sel,jel_a,jel_b,jel_c}.csv`) | regime, scenario, seeds, ΔFP×10³, FP/FN ratio, AUPRC, Recall@1% (mean ± std) | `scripts/run_baselines.py` + `scripts/summarize_results.py` | yes |
| `results/baselines_philadelphia_masked_m1check.csv` | one row per regime (single, finetune), 1 seed | `scripts/run_baselines.py --model masked` | yes |
| Paper reference values | SEL/JEL ΔFP×10³ and L_Rec(A′) from Table 2 | currently a constant in `app/api/main.py` quoting README.md; proposal: `results/paper_reference_philadelphia.json` (values checked by the report agent) | README only |
| `results/v1_<city>_scoring.csv` | city, seed, scorer, auprc, recall_at_1pct, delta_fp, fp_ratio, fn_ratio, train_seconds, score_seconds, epochs, buffer_size, buffer_rule, scenario | `scripts/run_v1.py` | yes (seeds 0-2, both cities) |
| `results/runs/v1_<city>_s<seed>.pt` (gitignored) | MaskedFeatureModel state_dict | `scripts/run_v1.py` | yes (s0-s2, both cities; local only) |
| `results/dashboard/<city>_alerts.json` | flat top-level keys: city, seed, scenario, model_file, experience, n_eval, budget, n_alerts, n_injected {global, local}, target_department, numeric_note, alerts, records; `alerts[scorer]` = [{rank, row, score, kind}]; `records[row]` = {department, kind, fields[{name, value, surprisal}], reason_codes} | `scripts/export_dashboard_data.py` (inference only, 26 s for both cities) | yes (seed 0, both cities; D1 done) |
| `results/strategies_<city>.csv` | city, seed, strategy, buffer_rule, auprc, recall_at_1pct, delta_fp, fp_ratio, fn_ratio, train_seconds | new: `scripts/run_baselines.py --model masked --regimes finetune --strategy ... --buffer-rule ...` writes one CSV per strategy today; a small aggregator or a `run_strategies.py` is needed | no (dependency D2, needs training runs) |
| `results/dashboard/<city>_buffer.json` | per rule: department share in the buffer after each experience | new export (hook into ReplayBuffer after each experience) | no (dependency D3, needs runs) |
| Elliptic results | per time step: AUPRC / recall on the delayed-label stream | not designed yet | no (blocked: Kaggle data) |

Dependencies (not run yet; need master approval):
- D1: run `scripts/export_dashboard_data.py` after the v1 run (inference only, light).
- D2: strategy comparison runs (masked, finetune, linear; none / ewc / replay×3 / ewc+replay; 3 seeds × 2 cities ≈ 6 configs × ~8 min × 6 runs, several hours of CPU; could start with 1 seed).
- D3: buffer-composition export (small additive hook, rides along with D2).
- D4: Elliptic real data (Kaggle login by a student).

## 4. Tech stack

FastAPI (read-only JSON over files in `configs/` and `results/`) + React 18 / Vite 5 / TypeScript / Tailwind 3 / Recharts 2,
as agreed. Reasons: the backend reuses Python and our CSV/JSON outputs directly and is testable with pytest
(`TestClient`); Vite gives a fast dev server with an `/api` proxy and a static `npm run build`; Recharts covers
stacked areas, grouped bars with error bars and horizontal bars, and has an `accessibilityLayer` option. No change proposed.
Versions are pinned exactly in `app/web/package.json`; `fastapi`/`uvicorn` are in `app/requirements.txt` only.

## 5. Design direction

- Layout: left sidebar (collapses to a top bar under 1024 px), one story step per page, cards on a 12-column grid,
  max width 1400 px. Every card carries its source file path.
- Typography: one sans family (Inter) for UI and a tabular-numeral setting for all numbers; one type scale
  (12 / 14 / 16 / 24 px). Per the frontend-design skill, avoid template tells: drop the tracked all-caps eyebrows and
  monospace labels used in the first scaffold, keep labels sentence case, keep lines under ~80 characters.
- Colour: dark theme (the user asked for a dark "control tower" look) but not "near-black + one neon accent"
  (flagged as generic by frontend-design): a slate-navy base, one signal colour for interaction, one alert colour
  reserved for the decaying department and for injected anomalies, and a colour-blind-safe categorical palette for the
  10 departments checked with the `dataviz` skill's validator. Contrast ≥ 4.5:1 for text (WCAG AA).
- Charts (dataviz skill): direct labels where possible, legends otherwise, light gridlines, error bars as ± std with
  the seed count stated, every chart paired with a table of the same numbers.
- Accessibility (web-design-guidelines + data-visualization-accessibility): semantic landmarks and headings, visible
  focus rings, keyboard navigation for nav, segmented controls and the record drawer (Esc closes, focus is trapped and
  restored), `aria-disabled` on M3–M5 entries, `prefers-reduced-motion` respected, Recharts `accessibilityLayer`.
- Copy: plain, short, states what each metric means once; empty states say which file is missing and which command
  makes it.

### 5.1 Design tokens (T1, as built)

Subject: an audit control tower watching a stream of city payments in ten time slices (E1–E10). The one memorable
element is the **experience runway**: a strip of ten cells E1→E10 under each page title, its cell heights showing the
target department shrinking, with E10 (the evaluated experience) lit like a runway threshold. Everything else stays
quiet: flat panels, hairlines, no gradients or shadows, no entrance animations.

Revisions against frontend-design's generic defaults: Inter → Barlow Semi Condensed (signage/airfield vernacular) for
titles and figures + IBM Plex Sans for UI; tracked all-caps eyebrows, monospace data labels and "→" links removed
(monospace stays only for shell commands); near-black + neon accent → deep navy with an amber "beacon" used for
interaction only; identical card kit → two radii (panel 10 px, control 6 px) and no shadows.

| Token | Hex | Role | Contrast (WCAG) |
|---|---|---|---|
| plane | `#0f1a2c` | page background | – |
| surface-1 | `#15233a` | panels, chart surface | – |
| surface-2 | `#1d2e4a` | raised: drawer, hover, selected row | – |
| line / line-strong | `#2b3d5e` / `#3a4e72` | hairlines, gridlines / axis, baseline | 1.45 / 1.88 on surface-1 (non-text) |
| ink | `#eef2f8` | primary text | 14.0 on surface-1, 12.1 on surface-2 |
| ink-2 | `#b4c0d4` | secondary text | 8.6 / 7.4 |
| ink-3 | `#8f9db5` | muted text, axis labels | 5.7 / 5.0 |
| beacon | `#f0b44c` | focus ring, active nav, links (never data) | 8.5 / 7.4 |
| series-1/2/3 | `#3987e5` / `#d95926` / `#199e70` | categorical, fixed order; Philadelphia = 1, Chicago = 2; emphasis = 1 | ≥ 3:1 marks (validator) |
| context-a/b | `#7d8ca8` / `#637390` | de-emphasised series (emphasis form); stacked segments separated by 2 px surface-coloured strokes | 4.64 / 3.29 (≥ 3:1, WCAG 1.4.11) |
| critical / critical-text | `#d03b3b` / `#f07a7a` | status: injected anomaly (icon + label) | mark 3.3; text 5.8 / 5.0 |
| good / good-text | `#0ca30c` / `#3fc46a` | status: good (icon + label) | mark 4.7; text 7.0 / 6.0 |

Type scale: 12 / 14 / 16 / 20 / 28 / 36 px; tabular figures only in tables and axes. Fonts are self-hosted from
pinned `@fontsource` packages (offline demo); no external font requests.

Status vs series (enforced): `critical` (`#d03b3b`) and Chicago's `series-2` (`#d95926`) are only 1.24:1 apart in
luminance with a similar hue, so (1) `critical` is never used as a mark inside a chart that also uses `series-2`;
(2) injected-anomaly marks appear only in the Alerts page and as the `InjectionLabel` component (icon + text label,
never colour alone); (3) alert tables and the record drawer show one city at a time and carry no city colour; (4) the
data-stream chart shows anomalies only in the table view, not as a coloured area. The beacon colour is reserved for
interaction and is not used for data (the runway's evaluated cell is drawn in primary ink).

Validator (`dataviz/scripts/validate_palette.js "#3987e5,#d95926,#199e70" --mode dark --surface "#15233a"`): lightness
band, chroma floor, CVD (worst adjacent ΔE 9.4 deutan), normal-vision floor (26.5) and contrast all PASS; `--pairs all`
also PASS (normal-vision 20.9). Charts carry at most 3 series.

## 6. Testing

- API: `tests/test_api.py` (exists, passes; skipped when fastapi is absent) on the real repo and on a temp root with
  fake files, covering pending states, parsing and 404s. Extend per new endpoint.
- Front end: `npm run typecheck` and `npm run build` must pass.
- Browser smoke tests with Python Playwright (webapp-testing skill pattern): start API + Vite, open each page, assert
  no console errors, the expected headings render, pending states render when files are absent, and the record drawer
  opens/closes with the keyboard; save screenshots to the scratchpad for review. Requires `pip install playwright` and
  `playwright install chromium` in py -3.12 (approval needed; large download).

## 7. Task breakdown

| # | Task | Deliverable | Depends on | Skill | Verification |
|---|---|---|---|---|---|
| T1 | Finish design tokens and revise the scaffold's visual language | `tailwind.config.js`, `index.css`, `palette.ts`; short token table in this doc | approval | frontend-design, dataviz | palette validator passes; contrast check |
| T2 | Shared components + accessibility pass | `components/ui.tsx` (card, stat, segmented, pending, error, source, chart+table) | T1 | web-design-guidelines, composition-patterns | typecheck; keyboard walk-through |
| T3 | Overview page (exists in draft) | `pages/Overview.tsx` | T2 | frontend-design | build; values match `summary_philadelphia_baseline_ae.md` |
| T4 | Data stream page (exists in draft) | `pages/Stream.tsx` + table fallback | T2 | dataviz | values match `experiences_*.csv` |
| T5 | Scoring page | `pages/Scoring.tsx` | T2, v1 run | dataviz | values match `v1_scoring_summary.md` |
| T6 | Alerts page + record drawer | `pages/Alerts.tsx` | T2, D1 | dataviz, web-design-guidelines | values match `<city>_alerts.json`; drawer keyboard test |
| T7 | Strategies page + `/api/strategies` | page, endpoint, test | T2, D2, D3 | dataviz | API test; pending state until D2 |
| T8 | Elliptic page + `/api/elliptic` (blocked state) | page, endpoint, test | T2 | — | pending state renders |
| T9 | Launcher | `scripts/run_dashboard.py` (starts uvicorn + `npm run dev`, Ctrl+C stops both) | — | — | both servers answer; `/api/health` via proxy |
| T10 | Repo hygiene | `.gitignore` += `app/web/node_modules/`, `app/web/dist/`, `__pycache__` already covered; README "Demo dashboard" section | T9 | — | `git status` clean of build output |
| T11 | Browser smoke tests | `tests/web/smoke_dashboard.py` (Playwright), screenshots | T3–T9 | webapp-testing | script passes locally |
| T12 | Paper reference values as data | `results/paper_reference_philadelphia.json`, API reads it | report agent confirms values | — | API test |

## 8. Current state (from the done-check)

Built before the process change, left in place: `app/api/main.py` (6 endpoints, tested), `app/requirements.txt`,
`app/web` scaffold (config, API client, UI components, App shell, Overview and Stream pages; Scoring and Alerts pages
missing, so `npm run build` does not pass yet), `scripts/export_dashboard_data.py` (not run), `tests/test_api.py`
(passes). No launcher, no README section, no `.gitignore` entries yet.
