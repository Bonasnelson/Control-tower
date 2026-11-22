"""Continual-learning strategies that plug into :class:`~controltower.training.loop.TrainHooks`.

* :class:`EWC` - Elastic Weight Consolidation (Kirkpatrick et al., 2017), in the
  "online" form (Schwarz et al., 2018) with decay 1: after every experience the
  diagonal Fisher of that experience is **added** to a running sum, and the anchor
  ``theta*`` is replaced by the parameters at the end of that experience. The
  penalty is ``lambda/2 * sum_i F_i (theta_i - theta*_i)^2``.
* :class:`ReplayBuffer` - a fixed-size memory of earlier records that is appended
  to every epoch of the next experiences. Three selection rules:
  ``random`` (reservoir sampling), ``diversity`` (greedy k-center on latent codes)
  and ``high_error`` (records with the highest anomaly score). No labels are used.
* :func:`combine_hooks` / :func:`make_hooks` - glue for ``run_regime(..., hooks=...)``.

The strategies are stateful: build a fresh one for every run.
"""
from __future__ import annotations

import numpy as np
import torch

from controltower.data.payments import TabularData
from controltower.models.autoencoder import OneHotAutoencoder
from controltower.training.loop import TrainHooks, tensors

STRATEGIES = ("none", "ewc", "replay", "ewc+replay")
BUFFER_RULES = ("random", "diversity", "high_error")


def _device(model: torch.nn.Module):
    return next(model.parameters()).device


# --------------------------------------------------------------------------- EWC
class EWC:
    """Diagonal-Fisher EWC penalty.

    Fisher estimate: the experience's records are subsampled (``n_samples``), split
    into mini-batches of ``batch_size``, and the squared gradient of ``model.loss``
    on each mini-batch is averaged over batches (the mini-batch empirical Fisher used
    by Avalanche's ``EWCPlugin``). The model is put in eval mode (no dropout) for
    the estimate. ``MaskedFeatureModel.loss`` draws a random field mask, so its
    Fisher estimate is itself stochastic (it averages over masks as well as records).
    Fishers are **summed** across experiences; the anchor is the latest snapshot.
    """

    def __init__(self, data: TabularData, lambda_: float, n_samples: int = 2048,
                 batch_size: int = 64, seed: int = 0):
        self.data, self.lambda_ = data, float(lambda_)
        self.n_samples, self.batch_size = n_samples, batch_size
        self.rng = np.random.default_rng(seed)
        self.fisher: dict[str, torch.Tensor] | None = None
        self.anchor: dict[str, torch.Tensor] | None = None

    def fisher_diag(self, model: torch.nn.Module, idx: np.ndarray) -> dict[str, torch.Tensor]:
        was_training = model.training
        model.eval()
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        fisher = {n: torch.zeros_like(p) for n, p in params.items()}
        sample = self.rng.permutation(idx)[: self.n_samples]
        n_batches = 0
        for s in range(0, len(sample), self.batch_size):
            cat, num = tensors(self.data, sample[s:s + self.batch_size], _device(model))
            model.zero_grad(set_to_none=True)
            model.loss(cat, num).backward()
            for n, p in params.items():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            n_batches += 1
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        return {n: f / max(n_batches, 1) for n, f in fisher.items()}

    def after_experience(self, model: torch.nn.Module, idx: np.ndarray) -> None:
        f = self.fisher_diag(model, idx)
        self.fisher = f if self.fisher is None else {n: self.fisher[n] + f[n] for n in f}
        self.anchor = {n: p.detach().clone() for n, p in model.named_parameters() if n in f}

    def penalty(self, model: torch.nn.Module) -> torch.Tensor:
        """Unscaled ``sum F (theta - theta*)^2`` (0 before the first experience)."""
        if self.fisher is None:
            return torch.zeros((), device=_device(model))
        total = torch.zeros((), device=_device(model))
        for n, p in model.named_parameters():
            if n in self.fisher:
                total = total + (self.fisher[n] * (p - self.anchor[n]) ** 2).sum()
        return total

    def extra_loss(self, model: torch.nn.Module) -> torch.Tensor:
        return 0.5 * self.lambda_ * self.penalty(model)

    def hooks(self) -> TrainHooks:
        return TrainHooks(extra_loss=self.extra_loss, after_experience=self.after_experience)


# ------------------------------------------------------------------------ replay
@torch.no_grad()
def embed(model: torch.nn.Module, data: TabularData, idx: np.ndarray, batch: int = 2048) -> np.ndarray:
    """Latent codes used by the ``diversity`` rule.

    ``MaskedFeatureModel.latent`` when available. ``OneHotAutoencoder`` has no
    ``latent`` method; it falls back to its bottleneck code (the encoder output,
    2-d by default). Any other model without ``latent`` raises ``TypeError``.
    """
    if hasattr(model, "latent"):
        fn = model.latent
    elif isinstance(model, OneHotAutoencoder):
        fn = lambda c, x: model(c, x)[1]  # noqa: E731  (x, z, x_hat) -> z
    else:
        raise TypeError(f"{type(model).__name__} has no .latent(); the diversity rule needs embeddings")
    was_training = model.training
    model.eval()
    out = [fn(*tensors(data, idx[s:s + batch], _device(model))).cpu().numpy()
           for s in range(0, len(idx), batch)]
    model.train(was_training)
    return np.concatenate(out).astype(np.float64)


@torch.no_grad()
def anomaly_scores(model: torch.nn.Module, data: TabularData, idx: np.ndarray, batch: int = 2048) -> np.ndarray:
    was_training = model.training
    model.eval()
    out = [model.score(*tensors(data, idx[s:s + batch], _device(model))).cpu().numpy()
           for s in range(0, len(idx), batch)]
    model.train(was_training)
    return np.concatenate(out)


def k_center_greedy(z: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Farthest-point (greedy k-center) selection of ``k`` rows of ``z``; random first point."""
    n = len(z)
    if k >= n:
        return np.arange(n)
    chosen = np.empty(k, dtype=np.int64)
    chosen[0] = rng.integers(n)
    d = ((z - z[chosen[0]]) ** 2).sum(1)
    for i in range(1, k):
        chosen[i] = int(np.argmax(d))
        d = np.minimum(d, ((z - z[chosen[i]]) ** 2).sum(1))
    return chosen


class ReplayBuffer:
    """Fixed-size replay memory refreshed after every experience.

    * ``random``: reservoir sampling over every record seen so far (each seen record
      is in the buffer with equal probability ``size / n_seen``).
    * ``diversity``: greedy k-center on the current model's latent codes of
      (buffer U candidates), where candidates are at most ``max_candidates`` records
      drawn at random from the new experience.
    * ``high_error``: keep the ``size`` records with the highest current anomaly
      score among (buffer U candidates). No labels are used, so the buffer holds the
      "hardest" records, which may include rare-but-normal records (and, in a real
      audit, undetected anomalies).
    """

    def __init__(self, data: TabularData, size: int, rule: str = "random", seed: int = 0,
                 max_candidates: int = 5000):
        if rule not in BUFFER_RULES:
            raise ValueError(f"rule must be one of {BUFFER_RULES}")
        self.data, self.size, self.rule = data, int(size), rule
        self.max_candidates = max_candidates
        self.rng = np.random.default_rng(seed)
        self.buffer = np.empty(0, dtype=np.int64)
        self.n_seen = 0

    def replay_indices(self) -> np.ndarray:
        return self.buffer

    def _reservoir(self, idx: np.ndarray) -> None:
        buf = list(self.buffer)
        for r in self.rng.permutation(idx):
            self.n_seen += 1
            if len(buf) < self.size:
                buf.append(r)
            else:
                j = self.rng.integers(self.n_seen)
                if j < self.size:
                    buf[j] = r
        self.buffer = np.asarray(buf, dtype=np.int64)

    def after_experience(self, model: torch.nn.Module, idx: np.ndarray) -> None:
        idx = np.asarray(idx, dtype=np.int64)
        if self.rule == "random":
            self._reservoir(idx)
            return
        self.n_seen += len(idx)
        new = self.rng.permutation(idx)[: self.max_candidates]
        pool = np.concatenate([self.buffer, new])
        if len(pool) <= self.size:
            self.buffer = pool
        elif self.rule == "diversity":
            self.buffer = pool[k_center_greedy(embed(model, self.data, pool), self.size, self.rng)]
        else:  # high_error
            s = anomaly_scores(model, self.data, pool)
            self.buffer = pool[np.argsort(-s, kind="stable")[: self.size]]

    def hooks(self) -> TrainHooks:
        return TrainHooks(replay_indices=self.replay_indices, after_experience=self.after_experience)


# -------------------------------------------------------------------------- glue
def combine_hooks(*strategies) -> TrainHooks:
    """Merge strategies (objects with optional ``extra_loss`` / ``replay_indices`` /
    ``after_experience``, or ``TrainHooks``): sums penalties, concatenates replay
    indices and calls every ``after_experience`` in order."""
    losses = [s.extra_loss for s in strategies if getattr(s, "extra_loss", None) is not None]
    replays = [s.replay_indices for s in strategies if getattr(s, "replay_indices", None) is not None]
    afters = [s.after_experience for s in strategies if getattr(s, "after_experience", None) is not None]

    def extra_loss(model):
        return sum((f(model) for f in losses[1:]), losses[0](model))

    def replay_indices():
        return np.concatenate([np.asarray(f(), dtype=np.int64) for f in replays])

    def after_experience(model, idx):
        for f in afters:
            f(model, idx)

    return TrainHooks(extra_loss=extra_loss if losses else None,
                      replay_indices=replay_indices if replays else None,
                      after_experience=after_experience if afters else None)


def make_hooks(strategy: str, cfg: dict, data: TabularData, seed: int = 0) -> TrainHooks | None:
    """Fresh hooks for ``strategy`` in ``STRATEGIES``; reads ``ewc_lambda``, ``ewc_samples``,
    ``buffer_size``, ``buffer_rule`` and ``buffer_candidates`` from ``cfg``."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if strategy == "none":
        return None
    parts = []
    if "ewc" in strategy:
        parts.append(EWC(data, cfg.get("ewc_lambda", 1.0), n_samples=cfg.get("ewc_samples", 2048), seed=seed))
    if "replay" in strategy:
        parts.append(ReplayBuffer(data, cfg.get("buffer_size", 2000), cfg.get("buffer_rule", "random"),
                                  seed=seed, max_candidates=cfg.get("buffer_candidates", 5000)))
    return combine_hooks(*parts)
