"""Elliptic Bitcoin dataset loader (Weber et al., 2019).

Expected files (Kaggle "ellipticco/elliptic-data-set", needs a Kaggle login), in
``data/elliptic/``:

* ``elliptic_txs_features.csv`` – no header; col 0 = txId, col 1 = time step (1..49),
  then 93 local + 72 aggregated features (165 total).
* ``elliptic_txs_classes.csv``  – header ``txId,class``; class "1" = illicit,
  "2" = licit, "unknown" = unlabelled.
* ``elliptic_txs_edgelist.csv`` – header ``txId1,txId2`` (payment flows).

The dark-market shutdown at time step 43 is the real drift event we evaluate on.

Done: loading, label mapping, time-step experiences, train-only standardisation
(Week 1); graph features from the edge list (``edge_features``), the delayed-label
predict-then-learn stream (``delayed_label_stream``) and an ``.npz`` cache
(``load_elliptic_cached``) (Week 2). Tested on synthetic CSVs only; the real Kaggle
files have not been run through this pipeline yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import scipy.sparse as sp

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


# ---------------------------------------------------------------- graph features
def edge_features(data: EllipticData) -> tuple[np.ndarray, list[str]]:
    """Per-node graph features from ``data.edges`` (needs ``with_edges=True``).

    Columns: ``in_degree`` and ``out_degree`` (edge counts, kept separate; an edge
    ``txId1 -> txId2`` is out for txId1 and in for txId2), then the mean of the
    **local** features (first 93 columns of ``X``) over the node's neighbours, with
    in- and out-neighbours **combined** into one undirected neighbour set (duplicate
    edges and a pair linked both ways count once; self-loops are ignored). Isolated
    nodes get 0 everywhere. Raw (unstandardised) features are averaged.
    Returns ``(F, names)``, ``F`` of shape (n, 2 + 93) float32. Sparse, O(n + m).
    """
    if data.edges is None:
        raise ValueError("edge_features needs the edge list: load with load_elliptic(..., with_edges=True)")
    n = len(data)
    src, dst = data.edges[:, 0].astype(np.int64), data.edges[:, 1].astype(np.int64)
    in_deg = np.bincount(dst, minlength=n)
    out_deg = np.bincount(src, minlength=n)
    keep = src != dst
    rows = np.concatenate([src[keep], dst[keep]])
    cols = np.concatenate([dst[keep], src[keep]])
    A = sp.csr_matrix((np.ones(len(rows), np.float64), (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    A.data[:] = 1.0  # binarise: each neighbour counts once
    n_nbr = np.asarray(A.sum(1)).ravel()
    mean = (A @ data.X[:, :N_LOCAL].astype(np.float64)) / np.maximum(n_nbr, 1)[:, None]
    F = np.column_stack([in_deg, out_deg, mean]).astype(np.float32)
    names = ["in_degree", "out_degree"] + [f"nbr_mean_local_{i}" for i in range(N_LOCAL)]
    return F, names


# ------------------------------------------------------------ delayed-label stream
@dataclass
class StreamStep:
    """One time step of the predict-then-learn protocol.

    The consumer first scores ``X`` (rows ``idx``, time step ``step``), then may learn
    from ``revealed_idx`` / ``revealed_y``: the licit/illicit labels that become known
    at this step. Evaluate predictions with ``data.y[idx]``; never train on it.
    """
    step: int
    idx: np.ndarray                  # rows of time step ``step``
    X: np.ndarray                    # (len(idx), d) float32, scaled with ``mu`` / ``sd``
    revealed_idx: np.ndarray         # labelled rows whose labels are released now
    revealed_y: np.ndarray           # their labels (1 illicit, 0 licit)
    revealed_steps: tuple[int, int]  # (lo, hi]: released rows come from steps lo < s <= hi
    mu: np.ndarray | None            # scaler statistics used (None if standardize=False)
    sd: np.ndarray | None
    scaler_rows: int                 # rows the scaler was fitted on (0 if standardize=False)


def delayed_label_stream(data: EllipticData, delay: int = 1, start_step: int = 1,
                         last_step: int | None = None, local_only: bool = False,
                         standardize: bool = True, eps: float = 1e-6) -> Iterator[StreamStep]:
    """Yield one :class:`StreamStep` per time step ``t = start_step .. last_step``.

    * Features: ``X_t`` is standardised with the mean/std of **all rows of steps < t**
      (features arrive at once, only labels are delayed). Warm-up: at the first time
      step of the data there is no history, so that step uses its own statistics
      (unsupervised, no labels); the scaler never sees a step > t.
    * Labels: at step ``t`` the labels of steps ``s <= t - delay`` not released before
      are released; in steady state that is exactly step ``t - delay``. The first
      yield releases the backlog ``s <= start_step - delay``. Only licit/illicit rows
      are released; unknown (-1) rows never are.
    """
    if delay < 0:
        raise ValueError("delay must be >= 0")
    X_all = data.X[:, :N_LOCAL] if local_only else data.X
    steps = data.time_step
    last_step = int(steps.max()) if last_step is None else int(last_step)
    d = X_all.shape[1]
    s1, s2, cnt = np.zeros(d), np.zeros(d), 0  # running sums over steps < t
    for t in range(int(steps.min()), start_step):
        Xs = X_all[steps == t].astype(np.float64)
        s1 += Xs.sum(0); s2 += (Xs ** 2).sum(0); cnt += len(Xs)
    lo = int(steps.min()) - 1  # nothing released yet
    for t in range(start_step, last_step + 1):
        idx = np.flatnonzero(steps == t)
        Xt = X_all[idx].astype(np.float64)
        mu = sd = None
        rows = 0
        if standardize:
            if cnt > 0:
                mu, rows = s1 / cnt, cnt
                sd = np.sqrt(np.maximum(s2 / cnt - mu ** 2, 0.0)) + eps
            else:  # warm-up: first step of the data uses its own statistics
                mu, sd, rows = Xt.mean(0), Xt.std(0) + eps, len(Xt)
            Xout = ((Xt - mu) / sd).astype(np.float32)
        else:
            Xout = Xt.astype(np.float32)
        hi = max(lo, t - delay)
        rev = np.flatnonzero((steps > lo) & (steps <= hi) & (data.y >= 0))
        yield StreamStep(step=t, idx=idx, X=Xout, revealed_idx=rev, revealed_y=data.y[rev],
                         revealed_steps=(lo, hi), mu=mu, sd=sd, scaler_rows=rows)
        lo = hi
        s1 += Xt.sum(0); s2 += (Xt ** 2).sum(0); cnt += len(Xt)


# ------------------------------------------------------------------------ caching
def _local(data: EllipticData) -> EllipticData:
    return EllipticData(tx_id=data.tx_id, time_step=data.time_step, X=data.X[:, :N_LOCAL],
                        y=data.y, feature_names=data.feature_names[:N_LOCAL], edges=data.edges)


def load_elliptic_cached(root: str | Path = "data/elliptic", cache: str | Path = "data/processed/elliptic.npz",
                         local_only: bool = False, with_edges: bool = False) -> EllipticData:
    """:func:`load_elliptic` with an ``.npz`` cache.

    The cache stores all 165 features (``local_only`` is applied after loading) and
    the edge list whenever it was parsed. It is used if it exists, is newer than every
    CSV present in ``root`` and holds edges when ``with_edges``; otherwise the CSVs are
    parsed and the cache is (re)written.
    """
    root, cache = Path(root), Path(cache)
    csvs = [root / f for f in ("elliptic_txs_features.csv", "elliptic_txs_classes.csv",
                               "elliptic_txs_edgelist.csv")]
    newest = max((p.stat().st_mtime for p in csvs if p.exists()), default=-np.inf)
    if cache.exists() and cache.stat().st_mtime > newest:
        with np.load(cache, allow_pickle=False) as z:
            if not with_edges or "edges" in z.files:
                data = EllipticData(tx_id=z["tx_id"], time_step=z["time_step"], X=z["X"], y=z["y"],
                                    feature_names=[str(s) for s in z["feature_names"]],
                                    edges=z["edges"] if with_edges else None)
                return _local(data) if local_only else data
    data = load_elliptic(root, local_only=False, with_edges=with_edges)
    cache.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(tx_id=data.tx_id, time_step=data.time_step, X=data.X, y=data.y,
                  feature_names=np.array(data.feature_names))
    if data.edges is not None:
        arrays["edges"] = data.edges
    tmp = cache.with_name(cache.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(cache)
    return _local(data) if local_only else data
