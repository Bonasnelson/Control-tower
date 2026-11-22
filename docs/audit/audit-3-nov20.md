# Audit 3: Week 2 milestone (due Fri Nov 20) and current state

Auditor: requirements auditor (read-only). Performed 2026-11-22, about 08:20, at `3a2ef4f`. Checks run: `pytest`
(59 passed), result-file comparisons and a code read.

## 1. Summary
W2-M was met two days late (`1d6198c`, Nov 22 04:13). v1 is trained on both cities with 3 seeds and 4 scorers,
Related Work is about 970 words, and the dashboard (T1–T12, D1–D3) is complete. No invented numbers are in the
report. R1–R5 have no FAIL, but all five are PARTIAL. The report has only Eq. 1 and one results table, and Elliptic
has no route to a model.

## 2. Plan vs reality
| Item | Status | Evidence |
|---|---|---|
| W2-C1 EWC and replay with 3 rules | Done | `training/strategies.py`, `tests/test_strategies.py` |
| W2-C2 Elliptic pipeline | Partial | Edge features, delayed-label stream and cache, tested on synthetic data. Real run blocked (#5) |
| W2-C3 four scorers | Done | `scoring.py`, `tests/test_scoring.py` |
| W2-C4 v1 and scoring comparison | Done | `results/v1_*_scoring.csv`, `v1_scoring_summary.md`/`.tex` |
| W2-R1 Figure 1 | Done | `report/figures/architecture.tex`. It matches `masked.py` and `train.yaml` (d=16, 256→128→32, ρ=0.3, F/K) |
| W2-R2, W2-R3 Related Work | Done | `main.tex` §2 |
| W2-R4 at least 15 references | Done | 19 entries, all cited |
| W2-M | Met (late) | `1d6198c`. Changelog #9 held the milestone until v1 was verified |
| Dashboard T1–T12, D1–D3 | Done | `047a4fe`…`3734e05`. D4 (Elliptic data) blocked |

Rules: every commit came from one git identity (the master). The code commits touch only allowed paths, and the
report commits (`c704aa6`, `2b2c70b`) touch only `report/`. The plan's single "Commit 1" was split into three
commits, which is harmless but unlogged. The D2 rerun of "replay random" repeats v1 seed 0, and it gives a
consistent AUPRC (0.730 against 0.731). The master plan's status table is still the original done-check and was
never updated.

## 3. Changes since the last audit
Changelog entries #15–#24 are all justified. #15 (1 seed for D2) weakens R3. #23 is good practice: measured data
took precedence over an instruction. Entry #18 sits out of order in the table, and #24 gives its reason as "–".

## 4. Requirements table
| Req | Status | Evidence | What to fix |
|---|---|---|---|
| R1 Novelty | PARTIAL | The claim is stated and scoped to the combination (§2, lines 158–163; novelty check). But the distinguishing parts (drift gate M3, feedback M4, canary M5, alert volume at equal recall) have no code | Rest the claim on what runs (see the strengthening ideas in the final message), and write the Intro contributions to match |
| R2 Approach | PARTIAL | Figure 1 and Eq. 1 are consistent with the code. Eq. 2–5 are `\todo`. `\tilde t` is undefined. Eq. 1 normalises per record, but the code divides by the mask count over the batch. The planned "Eq. 5 fused score" has no code. There are no equations for the buffer rules or the 4 scorers. Figure 2 is a `\todo` | Report agent: add the EWC (λ/2 Σ Fᵢ(θ−θ*)², Fisher summed), replay, k-center, high-error and 4-scorer equations. Rename the Fisher symbol so it does not clash with the field count F. Drop or implement "fused score" |
| R3 Ablations | PARTIAL | Scorers: 3 seeds, and field surprisal is best in both cities. Strategies: 1 seed, mixed results (EWC: Philadelphia AUPRC 0.77→0.61, Chicago 0.42→0.90; replay does not beat none). No masked vs one-hot AE comparison on the same footing. The report has no ablation table. Only E10 is evaluated, so forgetting is not measured | Code agent: run D2 with seeds 1–2, a one-hot AE SFT run, ewc+replay with all 3 rules, and a small λ sweep. Add a forgetting metric (score on E1–E9 held-out). Report agent: ablation table that states which choice each result supports or questions |
| R4 Datasets | PARTIAL | Philadelphia and Chicago: code and results (Chicago has v1 and D2 only, no SEL/JEL). Elliptic is blocked, and `MaskedFeatureModel([], K)` raises "stack expects a non-empty TensorList". No adapter exists from `EllipticData` to the training loop or metrics, and no plan task covers one | Code agent: numeric-only path and Elliptic run script, tested on the synthetic CSVs now. A student downloads the data. Report agent: per-dataset results |
| R5 Placement | PARTIAL | The Approach has no dataset section, but the Figure 1 legend gives per-city F/K. Datasets (§4.1) is a `\todo`. It needs the fields used, the deviations (Chicago 192k vs 399k rows, one-hot width 5,825 vs 5,588), the injection rules and the experience table | Report agent: move the F/K facts to §4.1, then write §4.1 completely |

## 5. Risks and gaps
- `fp_ratio` is 0 in every masked-model run, so it carries no information there. ΔFP is in nats for the masked model
  and in BCE for the AE, so the two cannot be compared across models.
- The `high_error` buffer after E10 holds all 20 injected anomalies in Philadelphia, and `diversity` holds 16. In a
  live stream these rules would replay fraud as normal data.
- Page count: about 1,600 words now (around 3 of the 12–14 pages). The abstract and Intro are `\todo`. No compile yet.

## 6. Required actions (through the master)
Code agent: the R3 runs, the R4 adapter, and a saved reason-code output. Report agent: the R2 equations, R5
§4.1, and the ablation and per-dataset tables once results exist. User: Overleaf compile and Kaggle download.
Master: update the status table and log the Week 3–4 tasks.
