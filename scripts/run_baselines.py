"""Reproduce the single-experience (SEL) and joint (JEL) baselines, and run the
fine-tuning (SFT) loop, for the baseline autoencoder or the masked-feature model.

    python scripts/run_baselines.py --city philadelphia --regimes single joint \
        --scenarios linear exponential instant --seeds 0 1 2 3 4
    python scripts/run_baselines.py --quick       # 2 epochs, 1 seed: smoke test

JEL is trained on the "full" scenario (no decay), as in the reference scripts.
Writes results/baselines_<city>_<model>.csv (one row per run) and a mean ± std
summary in results/baselines_<city>_<model>_summary.md.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from controltower.data.payments import build_experiences, load_or_prepare, load_yaml
from controltower.metrics import evaluate_last_experience
from controltower.training.loop import run_regime, score_records

ROOT = Path(__file__).resolve().parents[1]
COLS = ["delta_fp", "fp_ratio", "fn_ratio", "L_rec_target", "L_rec_local", "L_rec_global",
        "auprc", "recall_at_1pct", "train_seconds"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="philadelphia", choices=["philadelphia", "chicago"])
    ap.add_argument("--model", default="baseline_ae", choices=["baseline_ae", "masked"])
    ap.add_argument("--regimes", nargs="*", default=["single", "joint"],
                    choices=["single", "joint", "finetune"])
    ap.add_argument("--scenarios", nargs="*", default=["linear", "exponential", "instant"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=None, help="override configs/train.yaml")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="2 epochs, seed 0, linear only")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.quick:
        a.epochs, a.seeds, a.scenarios = 2, [0], ["linear"]
    torch.set_num_threads(a.threads)

    cfg = copy.deepcopy(load_yaml(ROOT / "configs" / "train.yaml")[a.model])
    if a.epochs is not None:
        cfg["epochs"] = a.epochs
    data = load_or_prepare(a.city, ROOT, seed=0)
    print(f"{a.city}: {len(data):,} records, one-hot dim {data.onehot_dim:,}, model={a.model}, cfg={cfg}")

    rows = []
    for regime in a.regimes:
        scenarios = ["full"] if regime == "joint" else a.scenarios
        for scen_name in scenarios:
            scen = load_yaml(ROOT / "configs" / "scenarios" / f"{scen_name}.yaml")
            for seed in a.seeds:
                exps = build_experiences(data, scen, np.random.default_rng(seed))
                model, hist = run_regime(regime, a.model, data, exps, cfg, seed)
                scores = score_records(model, data, exps[-1])
                res = evaluate_last_experience(scores, data, exps[-1])
                res.update(regime=regime, scenario=scen_name, seed=seed, model=a.model,
                           train_seconds=round(hist.seconds, 1),
                           final_train_loss=hist.epoch_loss[-1][-1], epochs=cfg["epochs"])
                print(f"[{regime:8s} {scen_name:11s} s{seed}] dFP={res['delta_fp']:+.5f} "
                      f"FP={res['fp_ratio']:.2f} FN={res['fn_ratio']:.2f} AUPRC={res['auprc']:.3f} "
                      f"R@1%={res['recall_at_1pct']:.2f} ({hist.seconds:.0f}s)", flush=True)
                res["per_dept"] = json.dumps(res["per_dept"])
                rows.append(res)

    df = pd.DataFrame(rows)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    stem = f"baselines_{a.city}_{a.model}{('_' + a.tag) if a.tag else ''}{'_quick' if a.quick else ''}"
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    summary = df.groupby(["regime", "scenario"])[COLS].agg(["mean", "std"])
    lines = [f"# Baselines: {a.city}, {a.model}, epochs={cfg['epochs']}, seeds={a.seeds}", "",
             "| regime | scenario | " + " | ".join(COLS) + " |",
             "|---|---|" + "---|" * len(COLS)]
    for (reg, scen), r in summary.iterrows():
        cells = [f"{r[(c, 'mean')]:.4g} ± {0 if np.isnan(r[(c, 'std')]) else r[(c, 'std')]:.2g}" for c in COLS]
        lines.append(f"| {reg} | {scen} | " + " | ".join(cells) + " |")
    (out_dir / f"{stem}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
