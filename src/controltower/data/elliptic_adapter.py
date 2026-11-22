"""Adapter from :class:`EllipticData` to the training loop, scorers and metrics.

The training loop and the scorers work on :class:`~controltower.data.payments.TabularData`
(integer categorical codes + numerical columns). Elliptic has only numerical features, so the
adapter builds a ``TabularData`` with **zero categorical fields** (``MaskedFeatureModel`` supports
that) and the (standardised) features as numerical fields.

Labels are used for evaluation only: ``group`` is ``GLOBAL_ANOMALY`` for illicit transactions and 0
otherwise (licit and unknown), so the models never see them during training. Evaluation is on the
labelled transactions of a time-step experience (illicit = positive class).

Standardisation uses the mean/std of the *first training experience only* (no look-ahead), and
clips standardised values to ±``clip`` so that a few extreme transactions cannot dominate the
squared-error loss.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from controltower.data.elliptic import EllipticData, edge_features
from controltower.data.payments import GLOBAL_ANOMALY, TabularData


def fit_standardizer(X: np.ndarray, idx: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    Xf = X[idx].astype(np.float64)
    return Xf.mean(0), Xf.std(0) + eps


def to_tabular(data: EllipticData, fit_idx: np.ndarray, with_edges: bool = False, clip: float = 10.0) -> TabularData:
    """Numerical-only ``TabularData`` for the training loop. ``fit_idx``: rows whose statistics
    standardise every row (use the first training experience)."""
    X, names = data.X.astype(np.float64), list(data.feature_names)
    if with_edges:
        F, fnames = edge_features(data)
        X, names = np.column_stack([X, F]), names + fnames
    mu, sd = fit_standardizer(X, fit_idx)
    Xs = np.clip((X - mu) / sd, -clip, clip).astype(np.float32)
    group = np.where(data.y == 1, GLOBAL_ANOMALY, 0).astype(np.int64)
    return TabularData(cat=np.zeros((len(data), 0), np.int64), num=Xs, group=group,
                       cat_names=[], num_names=names, cardinalities=[], vocab={},
                       dept_names=["transactions"], target_group=0)


def evaluate_labelled(scores: np.ndarray, y: np.ndarray, budget: float = 0.01) -> dict:
    """Metrics on one experience. ``y``: 1 illicit, 0 licit, -1 unknown (for the scored rows).

    * ``auprc``: average precision on the labelled rows only (illicit = positive);
    * ``recall_at_1pct``: share of the illicit rows among the top ``budget`` of *all* scored rows
      (the alert budget counts unlabelled transactions too, as an auditor would see them).
    """
    lab = y >= 0
    pos = y == 1
    res = {"n": int(len(y)), "n_illicit": int(pos.sum()), "n_licit": int((y == 0).sum()),
           "n_unknown": int((~lab).sum()), "auprc": float("nan"), "recall_at_1pct": float("nan")}
    if pos.any() and (y == 0).any():
        res["auprc"] = float(average_precision_score(pos[lab].astype(int), scores[lab]))
    if pos.any():
        k = max(1, int(np.ceil(budget * len(scores))))
        top = np.argsort(-scores)[:k]
        res["recall_at_1pct"] = float(pos[top].sum() / pos.sum())
    return res
