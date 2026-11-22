"""Forgetting metric: how well a continually trained model still fits earlier experiences.

A fixed sample of *normal* records (no injected anomalies) is held out from each of the first
experiences (E1..E9 by default) before training; those records are never trained on. After every
experience the model scores each held-out set, giving a matrix

    R[e, j] = mean anomaly score of held-out set j after training experience e     (lower = better fit)

Reported numbers (anomaly scores are "surprise", so lower means the experience is remembered):

* ``final_mean``: mean of the last row of R over all held-out sets (one number per run);
* ``final[j]``: last-row value per held-out set;
* ``forgetting[j]``: R[last, j] - min over e >= j of R[e, j], i.e. how much the score of set j rose
  after the best point reached once the model had seen experience j (0 = no forgetting);
* ``forgetting_mean``: mean of ``forgetting[j]``.

The score is ``model.score`` (field surprisal for the masked model, mean BCE for the one-hot AE),
the same score the E10 evaluation uses. Holding records out removes them from training, so runs with
the metric switched on are not bit-identical to runs without it; that is why the scripts keep it
behind a flag that is off by default.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from controltower.data.payments import GLOBAL_ANOMALY, TabularData
from controltower.training.loop import score_records


def holdout_split(data: TabularData, experiences: list[np.ndarray], per_experience: int = 200,
                  n_experiences: int | None = None, seed: int = 0) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return (training experiences, held-out sets). Held-out sets are drawn from the normal records
    of the first ``n_experiences`` experiences (default: all but the last) and removed from training."""
    n = len(experiences) - 1 if n_experiences is None else n_experiences
    rng = np.random.default_rng(seed)
    train, held = [], []
    for k, idx in enumerate(experiences):
        if k < n:
            normal = idx[data.group[idx] < GLOBAL_ANOMALY]
            h = np.sort(rng.choice(normal, size=min(per_experience, len(normal)), replace=False))
            held.append(h)
            train.append(np.setdiff1d(idx, h, assume_unique=False))
        else:
            train.append(idx)
    return train, held


@dataclass
class ForgettingTracker:
    """``on_experience(e, model)`` callback for ``run_regime``; fills the matrix R row by row."""
    data: TabularData
    holdouts: list[np.ndarray]
    device: str = "cpu"
    rows: list[list[float]] = field(default_factory=list)

    def on_experience(self, e: int, model: torch.nn.Module) -> None:
        was_training = model.training
        self.rows.append([float(score_records(model, self.data, h, device=self.device).mean()) for h in self.holdouts])
        model.train(was_training)

    @property
    def matrix(self) -> np.ndarray:
        return np.array(self.rows)

    def summary(self) -> dict:
        R = self.matrix
        if R.size == 0:
            return {"matrix": [], "final": [], "forgetting": [], "final_mean": float("nan"), "forgetting_mean": float("nan")}
        final = R[-1]
        forgetting = [float(final[j] - R[j:, j].min()) if j < len(R) else float("nan") for j in range(R.shape[1])]
        return {"matrix": R.tolist(), "final": final.tolist(), "forgetting": forgetting,
                "final_mean": float(final.mean()), "forgetting_mean": float(np.nanmean(forgetting)),
                "holdout_sizes": [int(len(h)) for h in self.holdouts]}


def chain(*callbacks):
    """Combine several ``on_experience`` callbacks (None entries are skipped)."""
    cbs = [c for c in callbacks if c is not None]
    if not cbs:
        return None

    def run(e, model):
        for c in cbs:
            c(e, model)
    return run
