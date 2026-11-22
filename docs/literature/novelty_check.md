# Novelty literature check (Week 1, about 1 hour)

**Question:** has anyone already built what we propose: a continual-learning anomaly detector
for ledger or transaction data that (a) uses embedding-based masked-feature reconstruction instead of
one-hot autoencoders, (b) separates legitimate drift from anomalies with a gate, (c) gives
per-field reason codes and uses analyst feedback, and (d) guards updates with a canary and rollback?

**Searches run (web, Nov 2026):** continual learning + anomaly detection + journal entries/audit + drift +
explainable; drift-aware CL + fraud + analyst feedback + replay; masked feature reconstruction + tabular
anomaly detection + continual; Elliptic + dark-market shutdown + continual learning.

## Closest prior work
| Work | What it does | What it lacks relative to us |
|---|---|---|
| Hemati, Schreyer, Borth 2022 (arXiv:2112.13215) | CL (SFT, EWC, ER) for one-hot AE on city payments; our baseline | no drift gate, no explanations, no feedback, no rollback; judged on reconstruction loss |
| Schreyer, Hemati, Borth, Vasarhelyi 2022 (arXiv:2210.15051) | Federated CL for accounting anomalies; local drift detection + clustering | same one-hot AE; no reason codes or analyst loop; FL is our "if time permits" item |
| Dalle Pezze et al. 2022 (arXiv:2212.11192) | CL approaches for anomaly detection (images) | not tabular, not financial |
| Thimonier et al., ICML 2024 (arXiv:2305.15121) | Masked-feature reconstruction for tabular AD (non-parametric transformer) | static, no CL, no drift handling |
| Dal Pozzolo et al., TNNLS 2018 | Fraud detection with concept drift, delayed labels, feedback loop | supervised, not unsupervised ledger AD; no CL or rollback |
| ROSFD 2025 (arXiv:2504.10229) | Streaming fraud detection; DDM/EDDM/ADWIN, ADWIN best | retrains on detection; no forgetting control or explanations |
| Preprint, Oct 2026 (preprints.org 202610.1810) | Splits analyst review budget between exploiting and exploring alerts; Elliptic replay | budget allocation, not model updating; not peer reviewed |
| Continual graph learning for AML review 2025 (arXiv:2503.24259) | Surveys CL on transaction graphs; Elliptic step 43 as case study | survey; graph-only |

## Conclusion
No single paper we found combines CL with a drift-versus-anomaly gate, per-field reason codes with feedback,
and canary-based rollback for journal-entry or transaction monitoring. The **combination** and the
**safety/governance evaluation** (alert volume at equal recall, rollback, E-23 framing) look novel.
Individual pieces are not new (EWC, replay, ADWIN, masked reconstruction), so the report must
claim novelty for the system and the evaluation, not for each component.

## Risks noted
- A 2026 DFRWS paper reports feature-semantics issues and train/test leakage in common Elliptic splits.
  Our Elliptic protocol must be strictly temporal (fit scalers on past steps only; the loader's
  `standardize` already does this).
- Label-free drift signals reportedly miss the step-43 shutdown (arXiv:2610.16534). That is a useful
  negative result to compare our M3 gate against.

## Sources
- https://arxiv.org/abs/2112.13215
- https://arxiv.org/abs/2210.15051
- https://arxiv.org/pdf/2212.11192
- https://arxiv.org/pdf/2305.15121
- https://boracchi.faculty.polimi.it/docs/2017_FraudsTNNLS.pdf
- https://arxiv.org/abs/2504.10229
- https://www.preprints.org/manuscript/202610.1810
- https://arxiv.org/pdf/2503.24259
- https://arxiv.org/pdf/2610.16534
- https://dfrws.org/wp-content/uploads/2026/04/DFRWS_USA_2026_Camera_Ready_Paper_65.pdf
- https://arxiv.org/pdf/2501.12723
