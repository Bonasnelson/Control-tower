"""Reason-code hit rate of model v1 on the injected anomalies of E10 (inference only).

For each saved v1 model (results/runs/v1_<city>_s<seed>.pt, written by scripts/run_v1.py) the
experiences are rebuilt with the same seed, every injected record in E10 is scored per field
(field surprisal: the masked model hides one field at a time), and its top-3 fields are compared
with the fields that inject_anomalies() (data/payments.py) changed:

* global anomalies: ``vendor_name``, the one categorical field set to a ``GLOBAL_RARE_*`` value
  (read back from the encoded vocabulary), and every numerical field (amount pushed to 2-5x the
  99.9th percentile);
* local anomalies: every categorical field except the department field (each copied from a record
  of another department); the amount is not changed.

The corrupted fields follow from the injection rules and the stored vocabulary, so no data
re-preparation is needed. Metrics per seed and anomaly kind:

* ``hit_any``: share of injected records whose top-3 contains at least one corrupted field;
* ``hit_share``: mean share of the top-3 that are corrupted fields (precision@3);
* ``chance_*``: the same quantities for 3 fields drawn uniformly at random (hypergeometric),
  as a baseline. For local anomalies at most 2 fields are not corrupted, so ``hit_any`` is 1 by
  construction; ``hit_share`` is the informative number there.

    python scripts/reason_code_hits.py --cities philadelphia chicago --seeds 0 1 2

Writes results/reason_codes_<city>.csv and results/reason_codes_summary.md.
"""
from __future__ import annotations

import argparse
import copy
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from controltower.data.payments import GLOBAL_ANOMALY, LOCAL_ANOMALY, build_experiences, load_or_prepare, load_yaml
from controltower.metrics import reason_codes
from controltower.training.loop import build_model, score_records

ROOT = Path(__file__).resolve().parents[1]
DEPT_FIELD = {"philadelphia": "dept", "chicago": "department_name"}
TOP = 3


def corrupted_fields(data, row: int, kind: int, dept_field: str) -> set[str]:
    if kind == GLOBAL_ANOMALY:
        out = {"vendor_name", *data.num_names}
        for j, c in enumerate(data.cat_names):
            if data.vocab[c][data.cat[row, j]].startswith("GLOBAL_RARE_"):
                out.add(c)
        return out
    return {c for c in data.cat_names if c != dept_field}


def chance(n_fields: int, n_corrupt: int, top: int = TOP) -> tuple[float, float]:
    """Random top-``top``: P(at least one corrupted), expected share corrupted."""
    p_none = comb(n_fields - n_corrupt, top) / comb(n_fields, top) if n_fields - n_corrupt >= top else 0.0
    return 1.0 - p_none, n_corrupt / n_fields


def run_city(city: str, seeds: list[int]) -> pd.DataFrame:
    cfg = copy.deepcopy(load_yaml(ROOT / "configs" / "train.yaml")["masked"])
    data = load_or_prepare(city, ROOT, seed=0)
    scen = load_yaml(ROOT / "configs" / "scenarios" / "linear.yaml")
    names = data.cat_names + data.num_names
    rows = []
    for seed in seeds:
        path = ROOT / "results" / "runs" / f"v1_{city}_s{seed}.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run scripts/run_v1.py first")
        exps = build_experiences(data, scen, np.random.default_rng(seed))
        model = build_model("masked", data, cfg)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        e10 = exps[-1]
        for kind, label in ((GLOBAL_ANOMALY, "global"), (LOCAL_ANOMALY, "local")):
            idx = e10[data.group[e10] == kind]
            codes = reason_codes(score_records(model, data, idx, per_field=True), names, top=TOP)
            any_hits, shares, n_corr = [], [], []
            for r, top3 in zip(idx, codes):
                bad = corrupted_fields(data, int(r), kind, DEPT_FIELD[city])
                hit = len(bad & set(top3))
                any_hits.append(hit > 0)
                shares.append(hit / TOP)
                n_corr.append(len(bad))
            ca, cs = zip(*(chance(len(names), k) for k in n_corr)) if n_corr else ((np.nan,), (np.nan,))
            rows.append({"city": city, "seed": seed, "kind": label, "n_records": len(idx), "n_fields": len(names),
                         "mean_corrupted_fields": float(np.mean(n_corr)) if n_corr else np.nan,
                         "hit_any": float(np.mean(any_hits)) if any_hits else np.nan,
                         "hit_share": float(np.mean(shares)) if shares else np.nan,
                         "chance_hit_any": float(np.mean(ca)), "chance_hit_share": float(np.mean(cs)),
                         "model_file": path.name, "scorer": "field_surprisal", "top": TOP})
            print(f"[{city} s{seed} {label}] n={len(idx)} hit_any={rows[-1]['hit_any']:.2f} "
                  f"hit_share={rows[-1]['hit_share']:.3f} (chance {rows[-1]['chance_hit_any']:.2f} / "
                  f"{rows[-1]['chance_hit_share']:.3f})", flush=True)
    return pd.DataFrame(rows)


def write_summary(out: Path, cities: list[str]) -> None:
    df = pd.concat([pd.read_csv(out / f"reason_codes_{c}.csv") for c in cities if (out / f"reason_codes_{c}.csv").exists()])
    cols = ["hit_any", "hit_share", "chance_hit_any", "chance_hit_share"]
    lines = ["# Reason-code hit rate, model v1 on E10 (field surprisal, top-3)", "",
             "hit_any: share of injected records whose top-3 fields include a corrupted field. hit_share: mean share "
             "of the top-3 that are corrupted fields. chance_*: the same for 3 random fields. Mean ± std over seeds. "
             "For local anomalies at most 2 fields are uncorrupted, so hit_any = 1 by construction; read hit_share.", "",
             "| city | kind | seeds | records per seed | " + " | ".join(cols) + " |", "|---|---|---|---|" + "---|" * len(cols)]
    for (c, k), g in df.groupby(["city", "kind"], sort=False):
        cells = [f"{g[m].mean():.3f} ± {g[m].std(ddof=1) if len(g) > 1 else 0:.3f}" for m in cols]
        lines.append(f"| {c} | {k} | {len(g)} | {g['n_records'].iloc[0]} | " + " | ".join(cells) + " |")
    (out / "reason_codes_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=["philadelphia", "chicago"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = ROOT / "results"
    for city in a.cities:
        run_city(city, a.seeds).to_csv(out / f"reason_codes_{city}.csv", index=False)
    write_summary(out, a.cities)


if __name__ == "__main__":
    main()
