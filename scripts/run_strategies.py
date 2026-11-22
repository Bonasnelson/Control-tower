"""Continual-strategy comparison for the masked-feature model (D2), with buffer composition (D3).

Sequential fine-tuning over the 10 experiences of a scenario (default "linear") with
each strategy: none (plain fine-tuning), ewc, replay with each buffer rule
(random / diversity / high_error) and ewc+replay (random buffer). Hyper-parameters
(epochs, ewc_lambda, buffer_size, ...) come from configs/train.yaml ``masked``.
Evaluation on the last experience E10 with the field-surprisal score
(``model.score``, the same score as scripts/run_baselines.py).

    python scripts/run_strategies.py --cities philadelphia --seeds 0
    python scripts/run_strategies.py --quick          # 2 epochs, both cities, seed 0
    ... --forgetting 200   # opt-in: hold out 200 normal records per E1..E9 and report forgetting
                           # (controltower/forgetting.py); outputs get a _forgetting suffix

Outputs (rows are appended after every run, so partial results survive):
* results/strategies_<city>.csv - one row per (strategy, buffer_rule, seed)
* results/dashboard/<city>_buffer.json - for every replay run: records per
  department (and injected-anomaly group) in the buffer after each experience
* results/strategies_summary.md - mean ± std over seeds per city and strategy
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from controltower.data.payments import GLOBAL_ANOMALY, LOCAL_ANOMALY, build_experiences, load_or_prepare, load_yaml
from controltower.metrics import evaluate_last_experience
from controltower.training.loop import run_regime, score_records
from controltower.training.strategies import EWC, ReplayBuffer, combine_hooks
from controltower.forgetting import ForgettingTracker, chain, holdout_split

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["auprc", "recall_at_1pct", "delta_fp", "fp_ratio", "fn_ratio"]
CONFIGS = [("none", None), ("ewc", None), ("replay", "random"), ("replay", "diversity"),
           ("replay", "high_error"), ("ewc+replay", "random")]


def group_name(data, g: int) -> str:
    return {GLOBAL_ANOMALY: "GLOBAL_ANOM", LOCAL_ANOMALY: "LOCAL_ANOM"}.get(int(g)) or data.dept_names[int(g)]


def run_one(data, exps, cfg, strategy, rule, seed, forgetting: int = 0):
    parts, buf = [], None
    if "ewc" in strategy:
        parts.append(EWC(data, cfg["ewc_lambda"], n_samples=cfg.get("ewc_samples", 2048), seed=seed))
    if "replay" in strategy:
        buf = ReplayBuffer(data, cfg["buffer_size"], rule, seed=seed, max_candidates=cfg.get("buffer_candidates", 5000),
                           filter_q=cfg.get("buffer_filter_q", 0.05))
        parts.append(buf)
    hooks = combine_hooks(*parts) if parts else None
    composition = []

    def on_experience(e, _model):
        if buf is not None:
            g = data.group[buf.replay_indices()]
            vals, counts = np.unique(g, return_counts=True)
            entry = {"experience": f"E{e + 1}", "size": int(len(g)),
                     "counts": {group_name(data, v): int(c) for v, c in zip(vals, counts)}}
            if rule == "filtered":  # what the anomaly filter removed at this refresh
                dropped = buf.last_dropped
                entry["dropped"] = int(len(dropped))
                entry["dropped_injected"] = int((data.group[dropped] >= GLOBAL_ANOMALY).sum())
            composition.append(entry)

    tracker = None
    if forgetting:  # optional (off by default): hold out normal records of E1..E9, never trained on
        exps, held = holdout_split(data, exps, per_experience=forgetting, seed=seed)
        tracker = ForgettingTracker(data, held)
    model, hist = run_regime("finetune", "masked", data, exps, cfg, seed, hooks=hooks,
                             on_experience=chain(on_experience, tracker.on_experience if tracker else None))
    res = evaluate_last_experience(score_records(model, data, exps[-1]), data, exps[-1])
    if tracker is not None:
        res["forgetting"] = tracker.summary()
    return res, hist, composition


def write_summary(out_dir: Path, cities: list[str]) -> None:
    frames = [pd.read_csv(p) for c in cities if (p := out_dir / f"strategies_{c}.csv").exists()]
    if not frames:
        return
    df = pd.concat(frames)
    df["buffer_rule"] = df["buffer_rule"].fillna("-")
    lines = ["# Continual strategies: masked model, fine-tuning, last experience E10 (field-surprisal score)", "",
             "Mean ± std over seeds (std 0 with one seed). Hyper-parameters from configs/train.yaml `masked`.", ""]
    for c in cities:
        d = df[df["city"] == c]
        if d.empty:
            continue
        lines += [f"## {c} (seeds {sorted(d['seed'].unique().tolist())}, epochs {d['epochs'].iloc[0]})", "",
                  "| strategy | buffer rule | " + " | ".join(METRICS) + " | train_seconds |",
                  "|---|---|" + "---|" * (len(METRICS) + 1)]
        for (s, r), g in d.groupby(["strategy", "buffer_rule"], sort=False):
            cells = [f"{g[m].mean():.4g} ± {0 if len(g) < 2 else g[m].std():.2g}" for m in METRICS + ["train_seconds"]]
            lines.append(f"| {s} | {r} | " + " | ".join(cells) + " |")
        lines.append("")
    (out_dir / "strategies_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=["philadelphia", "chicago"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--scenario", default="linear")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="2 epochs; outputs get a _quick suffix")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--configs", nargs="*", default=None, metavar="STRATEGY[:RULE]",
                    help="run only these configurations, e.g. replay:random replay:filtered (default: the 6 standard ones)")
    ap.add_argument("--append", action="store_true",
                    help="keep existing rows / buffer runs of other seeds in the output files and add this run's seeds")
    ap.add_argument("--forgetting", type=int, default=0, metavar="N",
                    help="hold out N normal records per experience E1..E9 and report forgetting (default 0 = off). "
                         "Outputs get a _forgetting suffix; the default outputs are not touched")
    a = ap.parse_args()
    out_dir = ROOT / "results"
    if a.summary_only:
        write_summary(out_dir, ["philadelphia", "chicago"])
        return
    if a.quick:
        a.epochs = 2
    # non-default scenario or configuration set -> separate output files, so the standard results stay untouched
    configs = CONFIGS if a.configs is None else [(c.split(":")[0], c.split(":")[1] if ":" in c else None) for c in a.configs]
    sfx = ((f"_{a.scenario}" if a.scenario != "linear" else "") + ("_forgetting" if a.forgetting else "")
           + ("_quick" if a.quick else ""))
    torch.set_num_threads(a.threads)
    cfg = copy.deepcopy(load_yaml(ROOT / "configs" / "train.yaml")["masked"])
    if a.epochs is not None:
        cfg["epochs"] = a.epochs
    scen = load_yaml(ROOT / "configs" / "scenarios" / f"{a.scenario}.yaml")
    for city in a.cities:
        data = load_or_prepare(city, ROOT, seed=0)
        csv_path = out_dir / f"strategies_{city}{sfx}.csv"
        buf_path = out_dir / "dashboard" / f"{city}_buffer{sfx}.json"
        buf_path.parent.mkdir(parents=True, exist_ok=True)
        buffers = {"city": city, "scenario": a.scenario, "buffer_size": cfg["buffer_size"],
                   "departments": data.dept_names, "target_department": data.dept_names[data.target_group],
                   "runs": []}
        rows, flog = [], []
        if a.append:  # keep earlier seeds (e.g. seed 0) and add the seeds of this run
            if csv_path.exists():
                old = pd.read_csv(csv_path, dtype={"buffer_rule": object})
                old = old[~old["seed"].isin(a.seeds)]
                rows = [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
                        for r in old.to_dict("records")]
            if buf_path.exists():
                prev = json.loads(buf_path.read_text(encoding="utf-8"))
                buffers["runs"] = [r for r in prev["runs"] if r["seed"] not in a.seeds]
            print(f"--append: kept {len(rows)} rows and {len(buffers['runs'])} buffer runs of other seeds", flush=True)
        print(f"{city}: {len(data):,} records, cfg={cfg}", flush=True)
        for seed in a.seeds:
            exps = build_experiences(data, scen, np.random.default_rng(seed))
            for strategy, rule in configs:
                res, hist, comp = run_one(data, exps, cfg, strategy, rule, seed, forgetting=a.forgetting)
                row = {"city": city, "seed": seed, "strategy": strategy, "buffer_rule": rule,
                       **{m: res[m] for m in METRICS}, "train_seconds": round(hist.seconds, 1),
                       "final_train_loss": hist.epoch_loss[-1][-1], "epochs": cfg["epochs"],
                       "ewc_lambda": cfg["ewc_lambda"] if "ewc" in strategy else None,
                       "buffer_size": cfg["buffer_size"] if rule else None, "scenario": a.scenario,
                       "scorer": "field_surprisal"}
                if comp:  # injected records in the buffers used for training (states after E1..E9)
                    inj = {c["experience"]: c["counts"].get("GLOBAL_ANOM", 0) + c["counts"].get("LOCAL_ANOM", 0)
                           for c in comp}
                    row.update({f"buffer_injected_after_{k}": inj.get(k, 0) for k in [f"E{i}" for i in range(5, 10)]})
                    row["buffer_injected_used_total"] = int(sum(v for k, v in inj.items() if k != comp[-1]["experience"]))
                    if rule == "filtered":
                        row["filter_dropped_injected_total"] = int(sum(c.get("dropped_injected", 0) for c in comp))
                if "forgetting" in res:
                    f = res["forgetting"]
                    row.update(forgetting_final_mean=f["final_mean"], forgetting_mean=f["forgetting_mean"],
                               forgetting_per_experience=a.forgetting)
                    flog.append({"city": city, "seed": seed, "strategy": strategy, "buffer_rule": rule, **f})
                    (out_dir / f"forgetting_strategies_{city}{sfx}.json").write_text(json.dumps(flog, indent=1), encoding="utf-8")
                rows.append(row)
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                if comp:
                    buffers["runs"].append({"strategy": strategy, "buffer_rule": rule, "seed": seed,
                                            "experiences": comp})
                    buf_path.write_text(json.dumps(buffers, indent=1), encoding="utf-8")
                print(f"[{city} s{seed} {strategy:10s} {rule or '-':10s}] AUPRC={row['auprc']:.3f} "
                      f"R@1%={row['recall_at_1pct']:.2f} dFP={row['delta_fp']:+.4f} FP={row['fp_ratio']:.2f} "
                      f"FN={row['fn_ratio']:.2f} ({hist.seconds:.0f}s)", flush=True)
    if not a.quick and not a.forgetting and a.scenario == "linear" and a.configs is None:
        write_summary(out_dir, ["philadelphia", "chicago"])


if __name__ == "__main__":
    main()
