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
scripts/                 download, prepare, run baselines
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
