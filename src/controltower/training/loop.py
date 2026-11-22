"""Training loop and learning regimes.

Regimes (Hemati et al., 2022, Sec. "Experimental setup"):

* ``single``   – Single Experience Learning (SEL): a fresh model trained only on the
  last experience E_N (today's common audit practice).
* ``joint``    – Joint Experience Learning (JEL): a fresh model trained on the union
  of all experiences E_1..E_N (the paper's lower bound on false positives).
* ``finetune`` – Sequential Fine-Tuning (SFT): one model trained on E_1, E_2, ...
  in order, with no regularisation or replay. This is the base continual loop;
  EWC and replay plug into it in Week 2 through the ``hooks`` argument.

As in Avalanche (which the reference code uses), the optimiser is re-created at
the start of every experience.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from controltower.data.payments import TabularData
from controltower.models.autoencoder import OneHotAutoencoder
from controltower.models.masked import MaskedFeatureModel

REGIMES = ("single", "joint", "finetune")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(kind: str, data: TabularData, cfg: dict) -> torch.nn.Module:
    if kind == "baseline_ae":
        return OneHotAutoencoder(data.cardinalities, data.num.shape[1],
                                 encoder_dim=cfg["encoder_dim"], bottleneck=cfg["bottleneck"])
    if kind == "masked":
        return MaskedFeatureModel(data.cardinalities, data.num.shape[1], d_embed=cfg["d_embed"],
                                  hidden=cfg["hidden"], latent=cfg["latent"],
                                  mask_ratio=cfg["mask_ratio"], dropout=cfg["dropout"])
    raise ValueError(f"unknown model kind {kind!r}")


def tensors(data: TabularData, idx: np.ndarray, device="cpu"):
    return (torch.as_tensor(data.cat[idx], device=device),
            torch.as_tensor(data.num[idx], device=device))


@dataclass
class TrainHooks:
    """Extension points for continual strategies (EWC penalty, replay sampling)."""
    extra_loss: Callable[[torch.nn.Module], torch.Tensor] | None = None
    replay_indices: Callable[[], np.ndarray] | None = None
    after_experience: Callable[[torch.nn.Module, np.ndarray], None] | None = None


@dataclass
class History:
    epoch_loss: list[list[float]] = field(default_factory=list)  # per experience
    seconds: float = 0.0


def train_experience(model, data: TabularData, idx: np.ndarray, cfg: dict, rng: np.random.Generator,
                     hooks: TrainHooks | None = None, device="cpu", patience: int | None = None,
                     tol: float = 1e-5) -> list[float]:
    """Train ``model`` on records ``idx`` for ``cfg['epochs']`` epochs with a fresh Adam.

    ``patience``: optional early stopping once the epoch loss stops improving by ``tol``.
    """
    hooks = hooks or TrainHooks()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    bs = cfg["batch_size"]
    losses, best, stale = [], np.inf, 0
    model.train()
    for _ in range(cfg["epochs"]):
        train_idx = idx
        if hooks.replay_indices is not None:
            train_idx = np.concatenate([idx, hooks.replay_indices()])
        order = rng.permutation(train_idx)
        total, n = 0.0, 0
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            cat, num = tensors(data, b, device)
            loss = model.loss(cat, num)
            if hooks.extra_loss is not None:
                loss = loss + hooks.extra_loss(model)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
            n += len(b)
        losses.append(total / max(n, 1))
        if patience is not None:
            if best - losses[-1] > tol:
                best, stale = losses[-1], 0
            else:
                stale += 1
                if stale >= patience:
                    break
    if hooks.after_experience is not None:
        hooks.after_experience(model, idx)
    model.eval()
    return losses


def run_regime(regime: str, kind: str, data: TabularData, experiences: list[np.ndarray],
               cfg: dict, seed: int, hooks: TrainHooks | None = None, device="cpu",
               on_experience: Callable[[int, torch.nn.Module], None] | None = None):
    """Train a fresh model under ``regime``; returns (model, History)."""
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}")
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(kind, data, cfg).to(device)
    hist, t0 = History(), time.perf_counter()
    if regime == "single":
        hist.epoch_loss.append(train_experience(model, data, experiences[-1], cfg, rng, hooks, device))
    elif regime == "joint":
        hist.epoch_loss.append(train_experience(model, data, np.concatenate(experiences), cfg, rng, hooks, device))
    else:  # finetune: sequential over experiences, parameters carried over
        for e, idx in enumerate(experiences):
            hist.epoch_loss.append(train_experience(model, data, idx, cfg, rng, hooks, device))
            if on_experience is not None:
                on_experience(e, model)
    hist.seconds = time.perf_counter() - t0
    return model, hist


@torch.no_grad()
def score_records(model, data: TabularData, idx: np.ndarray, device="cpu", per_field=False,
                  batch: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(idx), batch):
        cat, num = tensors(data, idx[s:s + batch], device)
        out.append((model.field_scores(cat, num) if per_field else model.score(cat, num)).cpu().numpy())
    return np.concatenate(out)
