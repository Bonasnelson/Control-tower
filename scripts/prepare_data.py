"""Encode a city dataset and print its 10-experience benchmark.

    python scripts/prepare_data.py --city philadelphia --scenario linear
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from controltower.data.payments import build_experiences, experience_summary, load_or_prepare, load_yaml

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=["philadelphia", "chicago"], nargs="*",
                    default=["philadelphia", "chicago"])
    ap.add_argument("--scenario", default="linear", choices=["linear", "exponential", "instant", "full"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    scen = load_yaml(ROOT / "configs" / "scenarios" / f"{a.scenario}.yaml")
    for city in a.city:
        data = load_or_prepare(city, ROOT, seed=0)
        exps = build_experiences(data, scen, np.random.default_rng(a.seed))
        print(f"\n=== {city}: {len(data):,} records (top-{len(data.dept_names)} depts + anomalies), "
              f"one-hot dim {data.onehot_dim:,}, target A' = {data.dept_names[data.target_group]}")
        print("cardinalities:", dict(zip(data.cat_names, data.cardinalities)))
        tab = experience_summary(data, exps)
        print(tab.to_string())
        out = ROOT / "results" / f"experiences_{city}_{a.scenario}.csv"
        out.parent.mkdir(exist_ok=True)
        tab.to_csv(out)
