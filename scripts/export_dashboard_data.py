"""Export per-record v1 outputs on the last experience E10 for the demo dashboard.

Loads a saved v1 model (results/runs/v1_<city>_s<seed>.pt, written by
scripts/run_v1.py), rebuilds the same experiences (same seed, "linear" scenario),
scores E10 with every scorer in ``controltower.scoring`` and writes
results/dashboard/<city>_alerts.json with:

* ``alerts[scorer]``: the top-1% records of E10 (alert budget of the metrics) with
  rank, score and whether the record is an injected global / local anomaly;
* ``records[row]``: for every alerted record, its department, decoded field values,
  per-field surprisal (``field_surprisal`` per field, i.e. the masked model hides
  each field in turn) and the top-3 reason codes.

Scoring is inference only (no training): a few seconds per city on a CPU.

    python scripts/export_dashboard_data.py --cities philadelphia chicago --seed 0
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from controltower.data.payments import GLOBAL_ANOMALY, LOCAL_ANOMALY, build_experiences, load_or_prepare, load_yaml
from controltower.metrics import reason_codes
from controltower.scoring import SCORERS, score
from controltower.training.loop import build_model

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 0.01


def kind_of(g: int) -> str:
    return {GLOBAL_ANOMALY: "global_anomaly", LOCAL_ANOMALY: "local_anomaly"}.get(int(g), "normal")


def export_city(city: str, seed: int, scenario: str = "linear") -> Path:
    model_path = ROOT / "results" / "runs" / f"v1_{city}_s{seed}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found: run scripts/run_v1.py first")
    cfg = copy.deepcopy(load_yaml(ROOT / "configs" / "train.yaml")["masked"])
    data = load_or_prepare(city, ROOT, seed=0)
    exps = build_experiences(data, load_yaml(ROOT / "configs" / "scenarios" / f"{scenario}.yaml"),
                             np.random.default_rng(seed))
    model = build_model("masked", data, cfg)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    train_idx, eval_idx = np.concatenate(exps[:-1]), exps[-1]
    n_alert = max(1, int(np.ceil(BUDGET * len(eval_idx))))
    g = data.group[eval_idx]

    alerts, alerted = {}, set()
    for name in SCORERS:
        s = score(name, model, data, train_idx, eval_idx, seed=seed)
        top = np.argsort(-s)[:n_alert]
        alerts[name] = [{"rank": r + 1, "row": int(eval_idx[j]), "score": float(s[j]), "kind": kind_of(g[j])}
                        for r, j in enumerate(top)]
        alerted |= {int(eval_idx[j]) for j in top}

    rows = np.array(sorted(alerted), dtype=np.int64)
    fs = score("field_surprisal", model, data, train_idx, rows, per_field=True)
    names = data.cat_names + data.num_names
    codes = reason_codes(fs, names)
    records = {}
    for i, r in enumerate(rows):
        grp = int(data.group[r])
        values = [data.vocab[c][data.cat[r, j]] for j, c in enumerate(data.cat_names)]
        values += [round(float(v), 6) for v in data.num[r]]
        records[str(int(r))] = {
            "row": int(r), "kind": kind_of(grp),
            "department": data.dept_names[grp] if grp < len(data.dept_names) else None,
            "fields": [{"name": n, "value": v, "surprisal": float(x)} for n, v, x in zip(names, values, fs[i])],
            "reason_codes": codes[i],
        }
    out = {"city": city, "seed": seed, "scenario": scenario, "model_file": model_path.name,
           "experience": "E10", "n_eval": int(len(eval_idx)), "budget": BUDGET, "n_alerts": n_alert,
           "n_injected": {"global": int((g == GLOBAL_ANOMALY).sum()), "local": int((g == LOCAL_ANOMALY).sum())},
           "target_department": data.dept_names[data.target_group],
           "numeric_note": "numerical fields are min-max scaled to [0, 1]",
           "alerts": alerts, "records": records}
    path = ROOT / "results" / "dashboard" / f"{city}_alerts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=["philadelphia", "chicago"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    for c in a.cities:
        print("wrote", export_city(c, a.seed))


if __name__ == "__main__":
    main()
