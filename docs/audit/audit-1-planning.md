# Audit 1: post-planning state

Auditor: requirements auditor (4th agent, read-only). Performed 2026-11-22, about 08:20, after the fact. Inputs: the
master plan file, `docs/dashboard_plan.md`, `docs/plan_changelog.md` entries 1 and 3–14, and commits `1fbb327` and `3245068`.

## 1. Summary
The Week 1–2 plan is concrete. Every W1/W2 item has an owner, a status and evidence, and the work order has
checkable steps. The dashboard plan (T1–T12, D1–D4) is equally concrete. The gap is requirement coverage: R1, R3
and R4 are only partly covered by tasks, and R2 equations, R5 and the Elliptic model run are covered by no task.

## 2. Plan vs reality (at planning time)
| Item group | Planned | Done | Partial | Not done |
|---|---|---|---|---|
| Week 1 (W1-C1…C5, R1, R2, S1, M) | 9 | C1–C5, R2, S1 | R1 (no template), M (fresh clone, compile) | – |
| Week 2 (W2-C1…C4, R1…R4, M) | 9 | R4 (19 refs) | C2, C3 | C1, C4, R1, R2, R3, M |
| Dashboard (T1–T12, D1–D4) | 16 | – | scaffold of T2–T4 (pre-plan) | rest |

| Requirement | Covered by plan task? |
|---|---|
| R1 novelty | W1-S1 novelty check; W2-R2/R3 each end with "how we differ". Nothing ties the claim to code that exists |
| R2 figures, equations | W2-R1 (Figure 1) only. No task for EWC, replay, buffer-rule or scoring equations, and none for Figure 2 |
| R3 ablations | W2-C4 scoring comparison only. The strategy ablation appears later, as dashboard dependency D2, not as an R3 task. No task for masked vs one-hot AE |
| R4 datasets | Philadelphia and Chicago: W1-C3, W2-C4. Elliptic: loader and stream only (W1-C5, W2-C2). No task to train or evaluate a model on it |
| R5 placement | No task; only a comment in `main.tex` (line 167) |

## 3. Changes since the last audit (none before this one)
| # | Change | Verdict |
|---|---|---|
| 1 | Done-check baseline | Justified |
| 3 | W1-R1 re-scoped to a template migration | Justified (template supplied mid-session) |
| 4 | Compile check moved to the user (Overleaf) | Justified, but it weakens W1-M and has no follow-up date |
| 5 | Elliptic tested on synthetic data only | Justified (Kaggle login). Weakens R4 |
| 6–9 | Commit policy, reorder, v1 definition, W2-M held until verified | Justified |
| 10–14 | Dashboard added, started early, plan-first process, skills, plan approved | Justified (user requests). #11 breaks the plan's own rule "starts only after the W2 check passes", but the change is logged |

## 4. Risks and gaps
- The master plan covers only Weeks 1–2. The Approach (Week 3) and Experiments (Week 4) work that R2, R3 and R5
  depend on exists only as `\todo`s in `main.tex`, with no owner or date in any plan.
- The dashboard T-tasks have no owner column (the code agent is implied), and D1–D4 have no owner. D4 needs a student.
- Elliptic: even after the download there is no planned path from `EllipticData` to a model or metric.
- The dashboard competes for time with the report. It is not a course requirement.

## 5. Required actions (assigned through the master)
- Master: add a Week 3–4 task list that maps each requirement to a task: R2 Eq. 2–6 and Figure 2 (report agent);
  R3 ablation runs and table (code agent, then report agent); R4 Elliptic adapter and run (code agent; data from a
  student); R5 Datasets section (report agent).
- Master: give D4 an owner (a student) and a date.
