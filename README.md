# Control Tower (DriftLedger)

A drift-aware, explainable continual-learning system for fraud and journal-entry risk monitoring
(COMP 5116 Machine Learning, Carleton University, Fall 2026, Bonaventure and Chang Lee).
The proposal is in `Control tower project.pdf` and the timeline in `Control tower project Schedule.docx`.

## Quick start (fresh clone)

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest                        # 12 tests on synthetic data, no download needed (~20 s)

python scripts/download_data.py         # Philadelphia FY2017 + Chicago payments -> data/raw/
python scripts/prepare_data.py          # encode + print the 10-experience benchmarks
python scripts/run_baselines.py --quick # 2-epoch smoke run of SEL/JEL
```

Tested with Python 3.12, PyTorch 2.6 (CPU), pandas 2.2 and scikit-learn 1.4.

## Reproducing the Hemati et al. (2022) baselines

```bash
# Single-experience (SEL), 3 decay scenarios x 5 seeds, 100 epochs
python scripts/run_baselines.py --city philadelphia --regimes single
# Joint (JEL), trained on the "full" scenario, 5 seeds (~25 min/seed on a CPU)
python scripts/run_baselines.py --city philadelphia --regimes joint
# Fine-tuning (SFT) or the masked-feature model
python scripts/run_baselines.py --regimes finetune
python scripts/run_baselines.py --model masked --regimes single joint finetune
```

Results are written to `results/baselines_<city>_<model>[_tag].csv`, with a mean ± std table in `*_summary.md`.

### Week 1 result: Philadelphia, one-hot AE, 5 seeds × 100 epochs (`results/summary_philadelphia_baseline_ae.md`)

| Regime | ΔFP ×10³ (ours) | ΔFP (paper, Table 2) | L_Rec(A') ×10³ ours / paper | FN ratio | AUPRC | Recall@1% |
|---|---|---|---|---|---|---|
| SEL (any decay) | +1.41 ± 0.04 | +1.45 / +1.40 / +1.45 | 6.26 / 6.59 | 0.98 ± 0.04 | 0.15 ± 0.02 | 0.24 ± 0.07 |
| JEL | −1.44 ± 0.06 | −1.36 ± 0.05 | 1.88 / 2.04 | 0.62 ± 0.13 | 0.36 ± 0.07 | 0.69 ± 0.07 |

SEL is identical across the linear, exponential and instant scenarios because it only trains on the last
experience, which is the same in all three (A' at 1%). The paper's small SEL differences are seed noise.
Local and global anomaly losses are lower than in the paper (JEL local 11.9 vs 36.7), so our injected anomalies are
somewhat harder. Run time: SEL ≈ 4 min per seed, JEL ≈ 40–75 min per seed on a 12-core CPU.

Preliminary M1 check (1 seed, 10 epochs, `results/baselines_philadelphia_masked_m1check.csv`): fine-tuning the
masked-feature model over the 10 experiences gives AUPRC 0.74 and Recall@1% 0.95. In
`scripts/check_reason_codes.py`, 70% of its top-3 reason codes on global anomalies are the corrupted fields.

### Week 2 result: model v1 scoring comparison (`results/v1_scoring_summary.md`)

v1 = masked-feature model (`configs/train.yaml` `masked`, 30 epochs), sequential fine-tuning over the 10 experiences
of the linear scenario with a 2,000-record random replay buffer; scored on E10, seeds 0, 1, 2
(`python scripts/run_v1.py`). The reference/fit set for latent distance and isolation forest is E1..E9.

| Scorer | Philadelphia AUPRC | Philadelphia Recall@1% | Chicago AUPRC | Chicago Recall@1% |
|---|---|---|---|---|
| Reconstruction error | 0.14 ± 0.06 | 0.25 ± 0.05 | 0.01 ± 0.00 | 0.02 ± 0.03 |
| Field surprisal | 0.67 ± 0.07 | 0.80 ± 0.05 | 0.59 ± 0.15 | 0.67 ± 0.19 |
| Latent k-NN distance | 0.23 ± 0.03 | 0.32 ± 0.06 | 0.25 ± 0.06 | 0.53 ± 0.08 |
| Isolation forest | 0.36 ± 0.07 | 0.53 ± 0.03 | 0.23 ± 0.06 | 0.43 ± 0.03 |

## Demo dashboard

A read-only web dashboard for the demo. It shows only numbers read from files in `results/`; anything not
computed yet shows the file it needs and the command that creates it.

```bash
python -m pip install -r app/requirements.txt   # fastapi + uvicorn (once)
cd app/web && npm ci && cd ../..                   # front-end packages, pinned in package-lock.json (once)
python scripts/run_dashboard.py                  # http://127.0.0.1:5173 ; Ctrl+C stops both servers
```

`run_dashboard.py` starts the API (uvicorn, port 8000) and the Vite dev server (port 5173, which proxies `/api`),
waits until `/api/health` answers through the proxy, and stops both on Ctrl+C. Options: `--open` (browser),
`--install` (run `npm ci` if needed), `--check` (start, check, stop), `--api-port`, `--web-port`, `--log-dir`.
Fonts are bundled, so it works offline.
On Windows with several Pythons installed, use `py -3.12` instead of `python` in these commands.

| Page | Shows | Reads |
|---|---|---|
| Overview | Week-1 baseline reproduction vs the paper; v1 scorer summary | `results/summary_philadelphia_baseline_ae.md`, `results/v1_<city>_scoring.csv` |
| Data stream | records per department E1–E10 (chart + table) | `results/experiences_<city>_linear.csv` |
| Scoring | v1 scorers: AUPRC and Recall@1% with ± std, all metrics | `results/v1_<city>_scoring.csv` |
| Strategies | none / EWC / replay (3 rules) / EWC+replay on E10 (seed 0), replay-buffer composition | `results/strategies_<city>.csv`, `results/dashboard/<city>_buffer.json` (`python scripts/run_strategies.py`) |
| Alerts | top-1% records on E10, per-field surprisal, reason codes | `results/dashboard/<city>_alerts.json` (`python scripts/export_dashboard_data.py`) |
| Elliptic | blocked until the Kaggle files are in `data/elliptic/`; what is ready | `data/elliptic/`, `src/controltower/data/elliptic.py` |

Checks: `python -m pytest -q -p no:cacheprovider tests/test_api.py` (API), and in `app/web`:
`npm run typecheck`, `npm run build`, and `node --experimental-strip-types --no-warnings checks/<page>_values.ts`
(with the API on port 8765) to compare every displayed value with its source file.

## Layout

```
configs/                 city configs, training hyper-parameters, decay scenarios (linear/exponential/instant/full)
src/controltower/
  data/payments.py       Philadelphia & Chicago: top-10 departments, anomaly injection, encoding, 10 experiences
  data/elliptic.py       Elliptic Bitcoin: loader, edge features, delayed-label stream, npz cache
                         (data from Kaggle; tested on synthetic CSVs only so far)
  models/autoencoder.py  baseline one-hot autoencoder (Hemati et al., 2022)
  models/masked.py       M1 masked-feature model (embeddings + masked reconstruction, per-field scores)
  training/loop.py       training loop; regimes single / joint / finetune; hooks for EWC & replay (Week 2)
  training/strategies.py EWC and replay buffer (random / diversity / high_error) as loop hooks
  metrics.py             dFP, FP/FN ratios, AUPRC, recall@1%, reason codes
scripts/                 download, prepare, run baselines / v1 / strategies, export + launch the dashboard
app/api/                 read-only FastAPI backend for the dashboard (serves configs/ and results/)
app/web/                 dashboard front end (React + Vite + TypeScript + Tailwind + Recharts)
tests/                   synthetic-data tests (fresh-clone check)
report/                  LaTeX report (10 pt, section skeleton) + references.bib
docs/meetings/           agendas + .ics invites; docs/literature/ novelty check
```

## Data

| Dataset | Source | Rows used |
|---|---|---|
| Philadelphia City Payments FY2017 | OpenDataPhilly (Carto table `city_payments_fy2017`) | 238,894 raw; 180,684 in the top-10 departments |
| Chicago Payments | data.cityofchicago.org `s4vu-giwb` (rows with a department) | 192,175 raw; 134,563 in the top-10 departments |
| Elliptic Bitcoin | Kaggle `ellipticco/elliptic-data-set` (manual download to `data/elliptic/`) | 203,769 tx |

## Deviations from the paper's setup (please read before comparing numbers)

1. **Encoded inputs are rebuilt.** The authors' encoded CSVs were never published (their README says "LINK-TO-BE-ADDED").
   Philadelphia uses `fm, dept, char_, sub_obj, vendor_name, doc_ref_no_prefix, contract_number` (one-hot) plus
   `transaction_amount` (min-max), which gives a one-hot dimension of 5,825 vs. 5,588 in the paper.
2. **Chicago has changed.** The portal now exposes only 6 columns, and department is filled for 192k rows
   (the paper used 399,158 rows and 66 categorical attributes). Contract numbers are capped at 2,000 values.
3. **Anomalies** follow the paper's definitions (global = rare values, local = unusual combinations),
   with 10 + 10 entering the last experience. We generate them without the Faker library.
4. **JEL is true joint training** on the union of experiences, as the paper defines it. The reference code
   calls Avalanche `Naive.train(stream)`, which trains the experiences one after another.
5. **Loss scale.** We report mean BCE per input dimension (the reference code's `BCELoss()`). The paper does not
   state its unit, but our values × 10³ land on the paper's scale (see below).
6. Departments with fewer than 10,000 records run out in experiences 7–10. This matches the reference code.
