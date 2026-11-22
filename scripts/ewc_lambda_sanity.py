"""EWC λ sanity check: size of the EWC penalty relative to the training loss (short run, < 10 min).

Philadelphia, linear scenario, seed 0, first 3 experiences, 3 epochs each, 4 threads. For each
setting the model is fine-tuned on E1, E2, E3; after each experience the diagonal Fisher is
added (training/strategies.py, EWC). Logged per experience: the mean training loss of its last
epoch, the unscaled penalty sum F (θ - θ*)^2 (mean over batches and at the end), the scaled
penalty λ/2 · sum, and the summed Fisher.

* masked, λ = 0      : free drift (penalty measured, not applied) -> what λ makes λ/2·penalty ≈ loss
* masked, λ = 1e4    : the configured value (configs/train.yaml), penalty applied
* baseline_ae, λ = 0 : free drift for the one-hot autoencoder

    python scripts/ewc_lambda_sanity.py | tee results/logs/ewc_lambda_sanity.log
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

from controltower.data.payments import build_experiences, load_or_prepare, load_yaml
from controltower.training.loop import TrainHooks, build_model, set_seed, train_experience
from controltower.training.strategies import EWC

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = [("masked", 0.0), ("masked", 1.0e4), ("baseline_ae", 0.0)]
EPOCHS, N_EXP, SEED = 3, 3, 0


def run(kind: str, lam: float, data, exps) -> list[dict]:
    cfg = copy.deepcopy(load_yaml(ROOT / "configs" / "train.yaml")[kind]) | {"epochs": EPOCHS}
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = build_model(kind, data, cfg)
    ewc = EWC(data, lambda_=lam, n_samples=cfg.get("ewc_samples", 2048), seed=SEED)
    pens: list[float] = []
    losses: list[float] = []
    base_loss = model.loss

    def logged_loss(cat, num):          # instance attribute: records the data loss of every batch
        out = base_loss(cat, num)
        losses.append(out.item())
        return out

    def extra(m):                       # records the unscaled penalty; applies λ/2 · penalty
        p = ewc.penalty(m)
        pens.append(p.item())
        return 0.5 * lam * p

    model.loss = logged_loss
    out = []
    for e in range(N_EXP):
        pens.clear(); losses.clear()
        t0 = time.perf_counter()
        train_experience(model, data, exps[e], cfg, rng, TrainHooks(extra_loss=extra), "cpu")
        t_train = time.perf_counter() - t0
        model.loss = base_loss
        t0 = time.perf_counter()
        ewc.after_experience(model, exps[e])
        t_fisher = time.perf_counter() - t0
        model.loss = logged_loss
        nb = max(1, len(losses) // EPOCHS)
        fsum = float(sum(f.sum().item() for f in ewc.fisher.values()))
        out.append({"kind": kind, "lambda": lam, "experience": f"E{e + 1}", "records": len(exps[e]),
                    "loss_last_epoch": float(np.mean(losses[-nb:])),
                    "penalty_mean": float(np.mean(pens)) if e else 0.0, "penalty_end": pens[-1] if e else 0.0,
                    "scaled_penalty_end": 0.5 * lam * (pens[-1] if e else 0.0), "fisher_sum": fsum,
                    "train_s": round(t_train, 1), "fisher_s": round(t_fisher, 2)})
        r = out[-1]
        print(f"{kind:11s} λ={lam:<7g} {r['experience']}: loss={r['loss_last_epoch']:.4g} "
              f"penalty mean={r['penalty_mean']:.3g} end={r['penalty_end']:.3g} "
              f"λ/2·end={r['scaled_penalty_end']:.3g} Fisher sum={fsum:.3g} ({r['train_s']}s train, {r['fisher_s']}s Fisher)",
              flush=True)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # λ in the log, also when piped on Windows
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    data = load_or_prepare("philadelphia", ROOT, seed=0)
    exps = build_experiences(data, load_yaml(ROOT / "configs" / "scenarios" / "linear.yaml"), np.random.default_rng(SEED))
    print(f"EWC λ sanity: philadelphia, linear, seed {SEED}, experiences E1-E{N_EXP} "
          f"({', '.join(str(len(e)) for e in exps[:N_EXP])} records), {EPOCHS} epochs each\n", flush=True)
    rows = [r for kind, lam in SETTINGS for r in run(kind, lam, data, exps)]
    print("\n| model | λ | exp | loss (last epoch) | penalty mean | penalty end | λ/2 · penalty end | Fisher sum |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['kind']} | {r['lambda']:g} | {r['experience']} | {r['loss_last_epoch']:.4g} | {r['penalty_mean']:.3g} | "
              f"{r['penalty_end']:.3g} | {r['scaled_penalty_end']:.3g} | {r['fisher_sum']:.3g} |")
    free = [r for r in rows if r["kind"] == "masked" and r["lambda"] == 0 and r["penalty_end"] > 0]
    if free:
        lam_eq = np.mean([2 * r["loss_last_epoch"] / r["penalty_end"] for r in free])
        print(f"\nmasked, free drift: λ that makes λ/2·penalty equal the loss at the end of E2/E3 ≈ {lam_eq:.3g}")
    print(f"total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
