"""Continual anomaly detection on Elliptic (numerical-only masked-feature model).

Time steps are grouped into experiences (default 5 steps each, so 49 steps -> 10 experiences).
The model is fine-tuned on experience k and then scores experience k+1 before seeing it
(predict-then-learn), so every row of the output is an evaluation on unseen, later data.
Labels (illicit / licit / unknown) are used for evaluation only.

    python scripts/run_elliptic.py --seeds 0 1 2                  # needs data/elliptic/*.csv (Kaggle)
    python scripts/run_elliptic.py --root <dir> --out-dir <dir> --epochs 2 --quick   # smoke run

Writes <out-dir>/elliptic_<tag>.csv (one row per seed x evaluated experience) and
<out-dir>/elliptic_<tag>_summary.md (mean ± std over seeds per evaluated experience).
The real Kaggle files are not in the repository; the script stops with a clear message if missing.
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from controltower.data.elliptic import SHUTDOWN_STEP, load_elliptic_cached, time_step_experiences
from controltower.data.elliptic_adapter import evaluate_labelled, to_tabular
from controltower.data.payments import load_yaml
from controltower.scoring import SCORERS, score
from controltower.training.loop import run_regime
from controltower.training.strategies import STRATEGIES, make_hooks

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "elliptic")
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "processed" / "elliptic.npz")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--steps-per-experience", type=int, default=5)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=None, help="override configs/train.yaml masked.epochs")
    ap.add_argument("--strategy", default="none", choices=STRATEGIES)
    ap.add_argument("--buffer-rule", default=None, choices=["random", "diversity", "high_error"])
    ap.add_argument("--scorer", default="field_surprisal", choices=SCORERS)
    ap.add_argument("--local-only", action="store_true", help="use the 93 local features only")
    ap.add_argument("--with-edges", action="store_true", help="append edge features (degrees, neighbour means)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true", help="smoke run: 1 seed; outputs get a _quick suffix")
    a = ap.parse_args()
    if a.quick:
        a.seeds = a.seeds[:1]
    torch.set_num_threads(a.threads)

    try:
        d = load_elliptic_cached(a.root, a.cache, local_only=a.local_only, with_edges=a.with_edges)
    except FileNotFoundError as e:
        print(f"Elliptic data not found: {e}\nDownload Kaggle 'ellipticco/elliptic-data-set' (login required) "
              f"and unzip the three CSV files into {a.root}.")
        return 1
    exps = [e for e in time_step_experiences(d, a.steps_per_experience, int(d.time_step.min()), int(d.time_step.max())) if len(e)]
    if len(exps) < 2:
        print("Need at least two non-empty experiences.")
        return 1
    data = to_tabular(d, exps[0], with_edges=a.with_edges)
    cfg = copy.deepcopy(load_yaml(ROOT / "configs" / "train.yaml")["masked"])
    if a.epochs is not None:
        cfg["epochs"] = a.epochs
    if a.buffer_rule:
        cfg["buffer_rule"] = a.buffer_rule
    print(f"Elliptic: {len(d):,} transactions, {data.num.shape[1]} numerical fields, {len(exps)} experiences "
          f"of {a.steps_per_experience} steps; strategy={a.strategy}, scorer={a.scorer}, epochs={cfg['epochs']}", flush=True)

    steps = [(int(d.time_step[e].min()), int(d.time_step[e].max())) for e in exps]
    rows = []
    for seed in a.seeds:
        t0 = time.perf_counter()

        def on_experience(k: int, model, seed=seed, t0=t0):
            if k + 1 >= len(exps):
                return
            ev = exps[k + 1]
            s = score(a.scorer, model, data, np.concatenate(exps[: k + 1]), ev, seed=seed)
            r = evaluate_labelled(s, d.y[ev])
            r.update(seed=seed, train_through=f"E{k + 1}", evaluated=f"E{k + 2}",
                     eval_steps=f"{steps[k + 1][0]}-{steps[k + 1][1]}",
                     after_shutdown=steps[k + 1][1] >= SHUTDOWN_STEP, seconds=round(time.perf_counter() - t0, 1))
            rows.append(r)
            print(f"[s{seed}] trained E1-E{k + 1} -> E{k + 2} (steps {r['eval_steps']}): AUPRC={r['auprc']:.3f} "
                  f"R@1%={r['recall_at_1pct']:.2f} illicit={r['n_illicit']} licit={r['n_licit']}", flush=True)

        run_regime("finetune", "masked", data, exps, cfg, seed, hooks=make_hooks(a.strategy, cfg, data, seed),
                   on_experience=on_experience)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    strat = a.strategy.replace("+", "-") + (f"_{cfg['buffer_rule']}" if "replay" in a.strategy else "")
    stem = f"elliptic_{strat}_{a.scorer}{'_' + a.tag if a.tag else ''}{'_quick' if a.quick else ''}"
    df = pd.DataFrame(rows).assign(strategy=a.strategy, scorer=a.scorer, epochs=cfg["epochs"],
                                   local_only=a.local_only, with_edges=a.with_edges)
    df.to_csv(a.out_dir / f"{stem}.csv", index=False)
    g = df.groupby(["evaluated", "eval_steps"], sort=False)[["auprc", "recall_at_1pct", "n_illicit"]].agg(["mean", "std"])
    lines = [f"# Elliptic, {a.strategy}, scorer {a.scorer}, seeds {a.seeds}, epochs {cfg['epochs']}", "",
             "Predict-then-learn: each row is scored by a model trained only on earlier experiences. "
             f"Dark-market shutdown at step {SHUTDOWN_STEP}.", "",
             "| evaluated | steps | AUPRC | Recall@1% | illicit |", "|---|---|---|---|---|"]
    for (ev, st), r in g.iterrows():
        f = lambda m: f"{r[(m, 'mean')]:.3f} ± {0 if np.isnan(r[(m, 'std')]) else r[(m, 'std')]:.3f}"  # noqa: E731
        lines.append(f"| {ev} | {st} | {f('auprc')} | {f('recall_at_1pct')} | {r[('n_illicit', 'mean')]:.0f} |")
    (a.out_dir / f"{stem}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {a.out_dir / (stem + '.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
