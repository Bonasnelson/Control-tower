"""Elliptic Bitcoin dataset loader (Weber et al., 2019) -- work in progress (Week 1 start).

Expected files (Kaggle "ellipticco/elliptic-data-set", needs a Kaggle login), in
``data/elliptic/``:

* ``elliptic_txs_features.csv`` – no header; col 0 = txId, col 1 = time step (1..49),
  then 93 local + 72 aggregated features (165 total).
* ``elliptic_txs_classes.csv``  – header ``txId,class``; class "1" = illicit,
  "2" = licit, "unknown" = unlabelled.
* ``elliptic_txs_edgelist.csv`` – header ``txId1,txId2`` (payment flows).

The dark-market shutdown at time step 43 is the real drift event we evaluate on.

Done: loading, label mapping, time-step experiences, train-only standardisation.
To do (Week 2, Chang Lee): edge features for the optional graph layer, the
delayed-label stream protocol, and caching.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_LOCAL = 93
N_AGG = 72
SHUTDOWN_STEP = 43
LABEL_MAP = {"1": 1, "2": 0, "unknown": -1}


@dataclass
class EllipticData:
    tx_id: np.ndarray        # (n,)
    time_step: np.ndarray    # (n,) 1..49
    X: np.ndarray            # (n, 165) float32
    y: np.ndarray            # (n,) 1 illicit, 0 licit, -1 unknown
    feature_names: list[str]
    edges: np.ndarray | None = None  # (m, 2) row indices into the arrays above

    def __len__(self) -> int:
        return len(self.y)


def feature_names() -> list[str]:
    return [f"local_{i}" for i in range(N_LOCAL)] + [f"agg_{i}" for i in range(N_AGG)]


def load_elliptic(root: str | Path = "data/elliptic", local_only: bool = False,
                  with_edges: bool = False) -> EllipticData:
    root = Path(root)
    feats_path = root / "elliptic_txs_features.csv"
    if not feats_path.exists():
        raise FileNotFoundError(
            f"{feats_path} not found. Download the Kaggle dataset 'ellipticco/elliptic-data-set' "
            f"and unzip the three CSV files into {root}/ (see README).")
    feats = pd.read_csv(feats_path, header=None)
    if feats.shape[1] != 2 + N_LOCAL + N_AGG:
        raise ValueError(f"expected {2 + N_LOCAL + N_AGG} columns, got {feats.shape[1]}")
    classes = pd.read_csv(root / "elliptic_txs_classes.csv", dtype={"class": str})
    label = feats[0].map(dict(zip(classes["txId"], classes["class"].map(LABEL_MAP))))
    if label.isna().any():
        raise ValueError(f"{int(label.isna().sum())} transactions have no class entry")

    names = feature_names()
    X = feats.iloc[:, 2:].to_numpy(np.float32)
    if local_only:
        X, names = X[:, :N_LOCAL], names[:N_LOCAL]
    data = EllipticData(tx_id=feats[0].to_numpy(np.int64), time_step=feats[1].to_numpy(np.int64),
                        X=X, y=label.to_numpy(np.int64), feature_names=names)
    if with_edges:
        e = pd.read_csv(root / "elliptic_txs_edgelist.csv")
        pos = pd.Series(np.arange(len(data)), index=data.tx_id)
        data.edges = np.stack([pos[e["txId1"]].to_numpy(), pos[e["txId2"]].to_numpy()], 1)
    return data


def time_step_experiences(data: EllipticData, steps_per_experience: int = 5,
                          first_step: int = 1, last_step: int = 49) -> list[np.ndarray]:
    """Group consecutive time steps into experiences (e.g. 49 steps -> 10 experiences of 5)."""
    out = []
    for start in range(first_step, last_step + 1, steps_per_experience):
        end = min(start + steps_per_experience - 1, last_step)
        out.append(np.flatnonzero((data.time_step >= start) & (data.time_step <= end)))
    return out


def standardize(train_X: np.ndarray, *others: np.ndarray, eps: float = 1e-6):
    """Fit mean/std on the training split only, apply to the others (no test leakage)."""
    mu, sd = train_X.mean(0), train_X.std(0) + eps
    return [(train_X - mu) / sd] + [(o - mu) / sd for o in others]


def summary(data: EllipticData) -> pd.DataFrame:
    df = pd.DataFrame({"step": data.time_step, "y": data.y})
    out = df.groupby("step")["y"].agg(
        n="size", illicit=lambda s: int((s == 1).sum()), licit=lambda s: int((s == 0).sum()))
    out["illicit_share_of_labelled"] = out["illicit"] / (out["illicit"] + out["licit"]).clip(lower=1)
    return out
