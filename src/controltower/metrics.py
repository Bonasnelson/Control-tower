"""Evaluation on the last (in-scope audit) experience.

Paper metrics (Hemati et al., 2022):
* ``L_rec[group]``     – mean reconstruction loss per department / anomaly type.
* ``delta_fp``         – L_rec(A') - max over the other departments of L_rec.
                         Negative: the decayed department A' is no longer the
                         worst-reconstructed one, i.e. fewer false positives.
* ``fp_ratio``         – reference-code FP: among normal records, take the top-k
                         scores with k = |A'| and report the share that belongs to A'.
* ``fn_ratio``         – reference-code FN: among normal + local anomalies, take the
                         top-k with k = #local anomalies; share that is NOT a local anomaly.
Added for the project's own protocol:
* ``auprc``            – average precision of anomalies (global + local) vs normal.
* ``recall_at_1pct``   – share of anomalies inside the top 1% scores (alert budget).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from controltower.data.payments import GLOBAL_ANOMALY, LOCAL_ANOMALY, TabularData


def evaluate_last_experience(scores: np.ndarray, data: TabularData, idx: np.ndarray,
                             budget: float = 0.01) -> dict:
    g = data.group[idx]
    target = data.target_group
    res: dict = {}
    normal_groups = [k for k in range(len(data.dept_names)) if (g == k).any()]
    per = {k: float(scores[g == k].mean()) for k in normal_groups}
    res["L_rec_target"] = per.get(target, np.nan)
    for k, name in ((GLOBAL_ANOMALY, "global"), (LOCAL_ANOMALY, "local")):
        res[f"L_rec_{name}"] = float(scores[g == k].mean()) if (g == k).any() else np.nan
    others = [v for k, v in per.items() if k != target]
    res["L_rec_max_other"] = max(others)
    res["delta_fp"] = res["L_rec_target"] - res["L_rec_max_other"]
    res["delta_fn"] = res["L_rec_target"] - res["L_rec_local"]

    normal = g < GLOBAL_ANOMALY
    k = int((g[normal] == target).sum())
    if k > 0:
        top = np.argsort(-scores[normal])[:k]
        res["fp_ratio"] = float((g[normal][top] == target).mean())
    else:
        res["fp_ratio"] = np.nan
    keep = g != GLOBAL_ANOMALY
    k = int((g[keep] == LOCAL_ANOMALY).sum())
    if k > 0:
        top = np.argsort(-scores[keep])[:k]
        res["fn_ratio"] = float((g[keep][top] != LOCAL_ANOMALY).mean())
    else:
        res["fn_ratio"] = np.nan

    y = (g >= GLOBAL_ANOMALY).astype(int)
    if y.any():
        res["auprc"] = float(average_precision_score(y, scores))
        n_alert = max(1, int(np.ceil(budget * len(scores))))
        top = np.argsort(-scores)[:n_alert]
        res["recall_at_1pct"] = float(y[top].sum() / y.sum())
    res["per_dept"] = {data.dept_names[k]: v for k, v in per.items()}
    return res


def reason_codes(field_scores: np.ndarray, field_names: list[str], top: int = 3) -> list[list[str]]:
    """Top-``top`` fields by reconstruction error for every record (M4 starting point)."""
    order = np.argsort(-field_scores, axis=1)[:, :top]
    return [[field_names[j] for j in row] for row in order]
