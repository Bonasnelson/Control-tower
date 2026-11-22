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

## Layout

```
configs/                 city configs, training hyper-parameters, decay scenarios (linear/exponential/instant/full)
src/controltower/
  data/payments.py       Philadelphia & Chicago: top-10 departments, anomaly injection, encoding, 10 experiences
  data/elliptic.py       Elliptic Bitcoin loader (started in Week 1; data from Kaggle)
  models/autoencoder.py  baseline one-hot autoencoder (Hemati et al., 2022)
  models/masked.py       M1 masked-feature model (embeddings + masked reconstruction, per-field scores)
  training/loop.py       training loop; regimes single / joint / finetune; hooks for EWC & replay (Week 2)
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
5. **Loss scale.** We report mean BCE per input dimension (the reference code's `BCELoss()`). The paper's tables are on a
   different, unstated scale, so compare signs and rankings, not absolute values.
6. Departments with fewer than 10,000 records run out in experiences 7–10. This matches the reference code.
