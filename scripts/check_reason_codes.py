"""Sanity check for per-field scores: for injected *global* anomalies we know which
fields were corrupted (vendor_name, one GLOBAL_RARE_* field, and the amount). Report how
often the model's top-3 reason codes contain them.

    python scripts/check_reason_codes.py --model masked --epochs 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from controltower.data.payments import GLOBAL_ANOMALY, build_experiences, load_or_prepare, load_yaml
from controltower.metrics import reason_codes
from controltower.training.loop import run_regime, score_records

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="philadelphia")
    ap.add_argument("--model", default="masked", choices=["masked", "baseline_ae"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    data = load_or_prepare(a.city, ROOT)
    cfg = load_yaml(ROOT / "configs" / "train.yaml")[a.model] | {"epochs": a.epochs}
    exps = build_experiences(data, load_yaml(ROOT / "configs" / "scenarios" / "linear.yaml"),
                             np.random.default_rng(a.seed))
    model, _ = run_regime("single", a.model, data, exps, cfg, a.seed)
    idx = exps[-1][data.group[exps[-1]] == GLOBAL_ANOMALY]
    names = data.cat_names + data.num_names
    codes = reason_codes(score_records(model, data, idx, per_field=True), names)
    hits = []
    for row, top3 in zip(idx, codes):
        corrupted = {"vendor_name", *data.num_names}
        for j, c in enumerate(data.cat_names):
            if data.vocab[c][data.cat[row, j]].startswith("GLOBAL_RARE_"):
                corrupted.add(c)
        hits.append(len(corrupted & set(top3)) / 3)
        print(f"top-3 {top3}  corrupted {sorted(corrupted)}")
    print(f"\nreason-code accuracy (share of top-3 that are corrupted fields): {np.mean(hits):.2f} "
          f"over {len(idx)} global anomalies")
