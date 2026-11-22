"""Anomaly scoring functions behind one interface (higher score = more anomalous).

    score(name, model, data, train_idx, eval_idx, *, seed=0, k=10, device="cpu",
          batch=2048, per_field=False) -> np.ndarray

* ``recon_error`` - one forward pass with every field visible, error summed over
  fields. For ``MaskedFeatureModel`` this is an all-False mask plus the model's
  ``_field_errors`` (cross-entropy per categorical field, squared error per
  numerical field): it measures how well the *visible* fields are copied through
  the bottleneck. ``field_surprisal`` instead hides each field and asks whether the
  other fields predict it. For ``OneHotAutoencoder`` it is ``model.score`` (mean BCE
  over the one-hot input, the paper's L_Rec); ``per_field`` gives ``field_scores``.
* ``field_surprisal`` - ``model.field_scores`` summed over fields (equals
  ``model.score`` for the masked model). ``per_field=True`` returns the (n, F)
  matrix, to feed :func:`controltower.metrics.reason_codes`. For the autoencoder
  ``field_scores`` is the per-field reconstruction error (no masking involved).
* ``latent_distance`` - mean Euclidean distance to the ``k`` nearest latent codes of
  a reference set: the codes of ``train_idx`` (seeded subsample of at most
  ``max_reference`` = 20,000 records). Embeddings as in the replay ``diversity``
  rule (``strategies.embed``): ``model.latent`` for the masked model, the bottleneck
  code for ``OneHotAutoencoder``, ``TypeError`` for any other model. Records that are
  also in the reference set count themselves (distance 0) among their neighbours.
* ``isolation_forest`` - model-free (``model`` may be ``None``). Features: relative
  frequency in ``train_idx`` of each categorical value (0 if unseen) plus the
  numerical fields; ``IsolationForest(n_estimators=200, contamination="auto",
  random_state=seed)`` fitted on ``train_idx``; score = ``-score_samples``.

``per_field`` is only defined for ``recon_error`` and ``field_surprisal``.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

from controltower.data.payments import TabularData
from controltower.models.masked import MaskedFeatureModel
from controltower.training.loop import score_records, tensors
from controltower.training.strategies import embed

SCORERS = ("recon_error", "field_surprisal", "latent_distance", "isolation_forest")
MAX_REFERENCE = 20_000


@torch.no_grad()
def _masked_recon_fields(model: MaskedFeatureModel, data: TabularData, idx: np.ndarray,
                         device="cpu", batch: int = 2048) -> np.ndarray:
    """(n, F) per-field error with no field masked."""
    model.eval()
    out = []
    for s in range(0, len(idx), batch):
        cat, num = tensors(data, idx[s:s + batch], device)
        mask = torch.zeros(cat.shape[0], model.n_fields, dtype=torch.bool, device=cat.device)
        _, logits, num_hat = model(cat, num, mask)
        out.append(model._field_errors(cat, num, logits, num_hat).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.n_fields), np.float32)


def recon_error(model, data, eval_idx, *, device="cpu", batch=2048, per_field=False) -> np.ndarray:
    if isinstance(model, MaskedFeatureModel):
        f = _masked_recon_fields(model, data, eval_idx, device, batch)
        return f if per_field else f.sum(1)
    return score_records(model, data, eval_idx, device=device, per_field=per_field, batch=batch)


def field_surprisal(model, data, eval_idx, *, device="cpu", batch=2048, per_field=False) -> np.ndarray:
    f = score_records(model, data, eval_idx, device=device, per_field=True, batch=batch)
    return f if per_field else f.sum(1)


def latent_distance(model, data, train_idx, eval_idx, *, seed=0, k=10, device="cpu", batch=2048,
                    max_reference: int = MAX_REFERENCE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ref = np.asarray(train_idx)
    if len(ref) > max_reference:
        ref = np.sort(rng.choice(ref, max_reference, replace=False))
    model.to(device)
    z_ref = embed(model, data, ref, batch)
    z_eval = embed(model, data, np.asarray(eval_idx), batch)
    nn_ = NearestNeighbors(n_neighbors=min(k, len(ref))).fit(z_ref)
    dist, _ = nn_.kneighbors(z_eval)
    return dist.mean(1)


def frequency_features(data: TabularData, train_idx: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Relative frequency (in ``train_idx``) of each categorical value, then the numerical fields."""
    cols = []
    for j, c in enumerate(data.cardinalities):
        freq = np.bincount(data.cat[train_idx, j], minlength=c) / max(len(train_idx), 1)
        cols.append(freq[data.cat[idx, j]])
    return np.column_stack(cols + [data.num[idx].astype(np.float64)])


def isolation_forest(data, train_idx, eval_idx, *, seed=0) -> np.ndarray:
    forest = IsolationForest(n_estimators=200, contamination="auto", random_state=seed)
    forest.fit(frequency_features(data, train_idx, train_idx))
    return -forest.score_samples(frequency_features(data, train_idx, eval_idx))


def score(name: str, model, data: TabularData, train_idx: np.ndarray, eval_idx: np.ndarray, *,
          seed: int = 0, k: int = 10, device="cpu", batch: int = 2048, per_field: bool = False) -> np.ndarray:
    """Score ``eval_idx`` with scorer ``name`` (see module docstring). ``train_idx`` is the
    reference/fit set for ``latent_distance`` and ``isolation_forest`` (ignored otherwise)."""
    if name not in SCORERS:
        raise ValueError(f"scorer must be one of {SCORERS}")
    train_idx, eval_idx = np.asarray(train_idx, dtype=np.int64), np.asarray(eval_idx, dtype=np.int64)
    if per_field and name not in ("recon_error", "field_surprisal"):
        raise ValueError(f"per_field is not defined for {name!r}")
    if name == "isolation_forest":
        return isolation_forest(data, train_idx, eval_idx, seed=seed)
    if model is None:
        raise ValueError(f"scorer {name!r} needs a trained model")
    if name == "recon_error":
        return recon_error(model, data, eval_idx, device=device, batch=batch, per_field=per_field)
    if name == "field_surprisal":
        return field_surprisal(model, data, eval_idx, device=device, batch=batch, per_field=per_field)
    return latent_distance(model, data, train_idx, eval_idx, seed=seed, k=k, device=device, batch=batch)
