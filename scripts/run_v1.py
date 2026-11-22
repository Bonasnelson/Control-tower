"""Model v1 and the first scoring comparison.

v1 = MaskedFeatureModel (configs/train.yaml ``masked``), sequential fine-tuning over
the 10 experiences of the "linear" scenario with a random-reservoir replay buffer
(``buffer_size`` from the config).

Evaluation on the last experience E10 for every scorer in ``controltower.scoring``.
The reference / fit set (``train_idx``) for ``latent_distance`` and
``isolation_forest`` is the union of the past experiences E1..E9 (the records the
model was trained on before E10); the other scorers only use the trained model.

    python scripts/run_v1.py --cities philadelphia chicago --seeds 0 1 2
    python scripts/run_v1.py --quick     # 2 epochs, seed 0: smoke test

Writes results/v1_<city>_scoring.csv (one row per seed x scorer; with --quick:
v1_<city>_scoring_quick.csv), and results/v1_scoring_summary{.md,.tex} built from
every results/v1_<city>_scoring.csv present (mean ± std over seeds). Model weights go
to results/runs/ (gitignored).
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from controltower.data.payments import build_experiences, load_or_prepare, load_yaml
from controltower.metrics import evaluate_last_experience
from controltower.scoring import SCORERS, score
from controltower.training.loop import run_regime
from controltower.training.strategies import make_hooks

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["auprc", "recall_at_1pct", "delta_fp", "fp_ratio", "fn_ratio"]
CITIES = ["philadelphia", "chicago"]
SCORER_LABEL = {"recon_error": "Reconstruction error", "field_surprisal": "Field surprisal",
                "latent_distance": "Latent $k$-NN distance", "isolation_forest": "Isolation forest"}


def run_city(city: str, seeds: list[int], cfg: dict, scenario: str, quick: bool) -> pd.DataFrame:
    data = load_or_prepare(city, ROOT, seed=0)
    scen = load_yaml(ROOT / "configs" / "scenarios" / f"{scenario}.yaml")
    print(f"{city}: {len(data):,} records, cfg={cfg}", flush=True)
    rows = []
    for seed in seeds:
        exps = build_experiences(data, scen, np.random.default_rng(seed))
        hooks = make_hooks("replay", cfg, data, seed)
        model, hist = run_regime("finetune", "masked", data, exps, cfg, seed, hooks=hooks)
        print(f"[{city} s{seed}] trained in {hist.seconds:.0f}s, final loss {hist.epoch_loss[-1][-1]:.4f}",
              flush=True)
        if not quick:
            run_dir = ROOT / "results" / "runs"
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), run_dir / f"v1_{city}_s{seed}.pt")
        train_idx, eval_idx = np.concatenate(exps[:-1]), exps[-1]
        for name in SCORERS:
            t0 = time.perf_counter()
            s = score(name, model, data, train_idx, eval_idx, seed=seed)
            secs = time.perf_counter() - t0
            res = evaluate_last_experience(s, data, eval_idx)
            row = {"city": city, "seed": seed, "scorer": name, **{m: res[m] for m in METRICS},
                   "train_seconds": round(hist.seconds, 1), "score_seconds": round(secs, 2),
                   "epochs": cfg["epochs"], "buffer_size": cfg["buffer_size"],
                   "buffer_rule": cfg["buffer_rule"], "scenario": scenario}
            rows.append(row)
            print(f"[{city} s{seed} {name:16s}] AUPRC={row['auprc']:.3f} R@1%={row['recall_at_1pct']:.2f} "
                  f"dFP={row['delta_fp']:+.4f} FP={row['fp_ratio']:.2f} FN={row['fn_ratio']:.2f} "
                  f"({secs:.1f}s)", flush=True)
    return pd.DataFrame(rows)


def _ms(x: pd.Series, fmt: str) -> tuple[str, str]:
    sd = x.std() if len(x) > 1 else 0.0
    return format(x.mean(), fmt), format(sd, fmt)


def write_summary(out_dir: Path, suffix: str = "") -> None:
    frames = [pd.read_csv(p) for c in CITIES if (p := out_dir / f"v1_{c}_scoring{suffix}.csv").exists()]
    if not frames:
        return
    df = pd.concat(frames)
    cities = [c for c in CITIES if c in set(df["city"])]
    cols = METRICS + ["score_seconds"]
    lines = ["# Model v1 scoring comparison (last experience E10, linear scenario)", "",
             "v1 = MaskedFeatureModel, sequential fine-tuning over 10 experiences, random replay buffer. "
             "Mean ± std over seeds; reference/fit set for latent distance and isolation forest = E1..E9.", ""]
    for c in cities:
        d = df[df["city"] == c]
        lines += [f"## {c} (seeds {sorted(d['seed'].unique().tolist())}, epochs {d['epochs'].iloc[0]}, "
                  f"train {d.groupby('seed')['train_seconds'].first().mean():.0f}s per seed)", "",
                  "| scorer | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for name in SCORERS:
            r = d[d["scorer"] == name]
            lines.append(f"| {name} | " + " | ".join(
                "{} ± {}".format(*_ms(r[m], ".4g" if m != "score_seconds" else ".1f")) for m in cols) + " |")
        lines.append("")
    (out_dir / f"v1_scoring_summary{suffix}.md").write_text("\n".join(lines), encoding="utf-8")
    tex = []
    for name in SCORERS:
        cells = []
        for c in cities:
            r = df[(df["city"] == c) & (df["scorer"] == name)]
            for m in ("auprc", "recall_at_1pct"):
                mean, sd = _ms(r[m], ".2f")
                cells.append(f"${mean} \\pm {sd}$")
        tex.append(f"{SCORER_LABEL[name]} & " + " & ".join(cells) + " \\\\")
    header = "% columns: scorer & " + " & ".join(f"{c} AUPRC & {c} Recall@1%" for c in cities)
    (out_dir / f"v1_scoring_summary{suffix}.tex").write_text(header + "\n" + "\n".join(tex) + "\n",
                                                             encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=CITIES, choices=CITIES)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--scenario", default="linear")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="2 epochs, seed 0")
    ap.add_argument("--summary-only", action="store_true", help="rebuild the summary from existing CSVs")
    a = ap.parse_args()
    out_dir = ROOT / "results"
    suffix = "_quick" if a.quick else ""
    if a.summary_only:
        write_summary(out_dir, suffix)
        return
    if a.quick:
        a.epochs, a.seeds = 2, [0]
    torch.set_num_threads(a.threads)
    cfg = copy.deepcopy(load_yaml(ROOT / "configs" / "train.yaml")["masked"]) | {"buffer_rule": "random"}
    if a.epochs is not None:
        cfg["epochs"] = a.epochs
    for city in a.cities:
        df = run_city(city, a.seeds, cfg, a.scenario, a.quick)
        df.to_csv(out_dir / f"v1_{city}_scoring{suffix}.csv", index=False)
    write_summary(out_dir, suffix)


if __name__ == "__main__":
    main()
