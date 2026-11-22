# Audit 2: Week 1 milestone (due Fri Nov 13)

Auditor: requirements auditor (read-only). Performed on 2026-11-22, nine days after the milestone date. It assesses
the state at `3245068` (Nov 22 03:00) plus the Week 1 items finished later (`c704aa6`, changelog #7).

## 1. Summary
The core Week 1 science is done and checks out. SEL and JEL reproduce the paper: ΔFP ×10³ is +1.41 ± 0.04 against
+1.40…1.45, and −1.44 ± 0.06 against −1.36 ± 0.05. Every number in Table 1 matches
`results/summary_philadelphia_baseline_ae.tex`. The milestone is still not fully met: nobody has confirmed the
template compiles, and the fresh-clone check left no stored evidence.

## 2. Plan vs reality
| Item | Status | Evidence |
|---|---|---|
| W1-C1 repo | Done | `1fbb327`, remote |
| W1-C2 M1 model and fine-tuning | Done | `models/masked.py`, `loop.py`, `results/baselines_philadelphia_masked_m1check.csv` (1 seed, 10 epochs) |
| W1-C3 two cities, 10 experiences | Done | `results/experiences_{philadelphia,chicago}_linear.csv` |
| W1-C4 SEL and JEL | Done | `results/summary_philadelphia_baseline_ae.md`, 5 seeds × 100 epochs |
| W1-C5 Elliptic loader | Done | `data/elliptic.py`, synthetic tests |
| W1-R1 template, 10pt | Partial | Migrated in `c704aa6`. The compile is unverified (moved to the user, #4) |
| W1-R2 at least 10 references | Done | 19 bib entries |
| W1-S1 novelty check | Done | `docs/literature/novelty_check.md` |
| W1-M milestone | Partial | Baselines: yes. Fresh clone: changelog #7 only (no log). Compile: open. Meeting notes for Nov 10 and 13 are blank |

Slippage: all Week 1 work is dated Nov 22, nine days late. The cause, per changelog #2, is a late start; there are no
earlier commits. Rules: `1fbb327` and `3245068` mix `report/` with code paths, but they predate the agent
file-ownership rule, so this is noted, not flagged.

## 3. Changes since the last audit
| # | Change | Verdict |
|---|---|---|
| 2 | Milestones slipped | Reason given, but the Agent column is "–". Justified. Weakens both milestones |
| 7 | Fresh-clone check run in parallel with W1-R1 | Justified. No result recorded |
| (unlogged) | `2b2c70b` corrects the Table 1 caption: the ×10³ unit is ours, because the paper states none | Justified fix. It is a correction, not a plan change |

## 4. Risks and gaps
- Numbers without a stored output:
  - The README's "70% of top-3 reason codes are the corrupted fields" (`scripts/check_reason_codes.py`) is printed
    only, never saved. The script also trains `single`, not the fine-tuned model the README sentence implies.
  - The `train.yaml` comment "sanity run … λ/2·penalty ≈ loss" has no log.
  - Neither number is in the report yet. Both must not enter it without a file.
- The M1 check is 1 seed and 10 epochs, and the one-hot AE was never run under fine-tuning (SFT). Without that
  run there is no like-for-like masked vs one-hot comparison (R3).
- The `masked.py` docstring says "5,588-wide one-hot", which is the paper's width. Our width is 5,825 (README).

## 5. Required actions (through the master)
- User: compile on Overleaf and confirm 10pt. Master: log the result.
- Code agent: rerun the fresh-clone check and save its output under `results/logs/`. Make
  `check_reason_codes.py` write a CSV, and run it on the fine-tuned model.
- Master: mark W1-M met only when both of the above are recorded.
