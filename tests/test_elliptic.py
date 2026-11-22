"""Elliptic pipeline on tiny synthetic CSVs (the real Kaggle files are not needed)."""
import os

import numpy as np
import pandas as pd
import pytest

from controltower.data import elliptic as el

N_STEPS, PER_STEP = 6, 20


def write_fake(root, edges=((0, 1), (2, 1), (1, 3), (3, 1), (4, 4), (0, 1)), seed=0):
    """6 time steps x 20 tx; txIds are 1000 + row. ``edges`` are row-index pairs."""
    rng = np.random.default_rng(seed)
    n = N_STEPS * PER_STEP
    tx = 1000 + np.arange(n)
    step = np.repeat(np.arange(1, N_STEPS + 1), PER_STEP)
    # feature value encodes the step, so leakage of a later step shows up in the scaler
    X = rng.normal(size=(n, el.N_LOCAL + el.N_AGG)) + step[:, None] * 10.0
    pd.DataFrame(np.column_stack([tx, step, X])).to_csv(root / "elliptic_txs_features.csv",
                                                        header=False, index=False)
    cls = np.array(["1", "2", "unknown"])[rng.integers(0, 3, n)]
    pd.DataFrame({"txId": tx, "class": cls}).to_csv(root / "elliptic_txs_classes.csv", index=False)
    e = np.asarray(edges)
    pd.DataFrame({"txId1": tx[e[:, 0]], "txId2": tx[e[:, 1]]}).to_csv(
        root / "elliptic_txs_edgelist.csv", index=False)
    return X, step, cls


@pytest.fixture
def root(tmp_path):
    write_fake(tmp_path)
    return tmp_path


def test_load(root):
    d = el.load_elliptic(root, with_edges=True)
    assert d.X.shape == (120, 165) and d.time_step.min() == 1 and d.time_step.max() == 6
    assert set(np.unique(d.y)) <= {-1, 0, 1}
    assert d.edges.shape == (6, 2) and d.edges[0].tolist() == [0, 1]
    assert el.load_elliptic(root, local_only=True).X.shape == (120, 93)


def test_edge_features_hand_checked(root):
    d = el.load_elliptic(root, with_edges=True)
    F, names = el.edge_features(d)
    assert F.shape == (120, 2 + 93) and len(names) == 95 and names[:2] == ["in_degree", "out_degree"]
    # edges: 0->1 (twice), 2->1, 1->3, 3->1, 4->4 (self-loop)
    assert F[:5, 0].tolist() == [0, 4, 0, 1, 1]  # in-degree (edge counts)
    assert F[:5, 1].tolist() == [2, 1, 1, 1, 1]  # out-degree
    L = d.X[:, :93].astype(np.float64)
    np.testing.assert_allclose(F[1, 2:], L[[0, 2, 3]].mean(0), rtol=1e-5)  # unique nbrs of 1: {0,2,3}
    np.testing.assert_allclose(F[0, 2:], L[1], rtol=1e-5)
    np.testing.assert_allclose(F[3, 2:], L[1], rtol=1e-5)  # 1<->3 counted once
    assert (F[4, 2:] == 0).all()                           # self-loop only: no neighbours
    assert (F[5:] == 0).all()                              # isolated nodes
    with pytest.raises(ValueError, match="with_edges"):
        el.edge_features(el.load_elliptic(root))


@pytest.mark.parametrize("delay", [0, 1, 2])
def test_stream_no_leakage(root, delay):
    d = el.load_elliptic(root)
    seen_labels = set()
    out = list(el.delayed_label_stream(d, delay=delay))
    assert [s.step for s in out] == list(range(1, 7))
    for s in out:
        assert (d.time_step[s.idx] == s.step).all() and len(s.idx) == PER_STEP
        past = d.time_step < s.step
        if s.step == 1:  # documented warm-up: own statistics
            assert s.scaler_rows == PER_STEP
            np.testing.assert_allclose(s.mu, d.X[s.idx].mean(0), rtol=1e-6)
        else:            # only rows of steps < t
            assert s.scaler_rows == past.sum()
            np.testing.assert_allclose(s.mu, d.X[past].astype(np.float64).mean(0), rtol=1e-6)
            np.testing.assert_allclose(s.sd, d.X[past].astype(np.float64).std(0) + 1e-6, rtol=1e-5)
            np.testing.assert_allclose(s.X, (d.X[s.idx] - s.mu) / s.sd, rtol=1e-4, atol=1e-4)
        # labels: only from steps <= t - delay, licit/illicit only, each released once
        assert (d.time_step[s.revealed_idx] <= s.step - delay).all()
        assert (s.revealed_y >= 0).all() and (s.revealed_y == d.y[s.revealed_idx]).all()
        assert not seen_labels & set(s.revealed_idx.tolist())
        seen_labels |= set(s.revealed_idx.tolist())
        if s.step - delay >= 1:  # steady state: exactly step t - delay
            assert set(d.time_step[s.revealed_idx]) <= {s.step - delay}
            assert len(s.revealed_idx) == ((d.time_step == s.step - delay) & (d.y >= 0)).sum()
    labelled_by_end = (d.y >= 0) & (d.time_step <= 6 - delay)
    assert seen_labels == set(np.flatnonzero(labelled_by_end).tolist())


def test_stream_late_start_and_options(root):
    d = el.load_elliptic(root)
    first = next(el.delayed_label_stream(d, delay=1, start_step=4, local_only=True))
    assert first.X.shape == (PER_STEP, 93) and first.scaler_rows == 3 * PER_STEP
    assert set(d.time_step[first.revealed_idx]) <= {1, 2, 3}  # backlog s <= 3
    assert len(first.revealed_idx) == ((d.time_step <= 3) & (d.y >= 0)).sum()
    raw = next(el.delayed_label_stream(d, standardize=False))
    assert raw.mu is None and np.allclose(raw.X, d.X[raw.idx])


def test_cache_round_trip(root, tmp_path):
    cache = tmp_path / "proc" / "elliptic.npz"
    a = el.load_elliptic_cached(root, cache, with_edges=True)
    assert cache.exists()
    os.remove(root / "elliptic_txs_features.csv")  # second load must come from the cache
    b = el.load_elliptic_cached(root, cache, with_edges=True)
    for f in ("tx_id", "time_step", "X", "y", "edges"):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))
    assert a.feature_names == b.feature_names
    c = el.load_elliptic_cached(root, cache, local_only=True)
    assert c.X.shape == (120, 93) and c.edges is None


def test_cache_refreshes_when_csv_newer(tmp_path):
    root = tmp_path / "raw"; root.mkdir()
    write_fake(root, seed=0)
    cache = tmp_path / "elliptic.npz"
    a = el.load_elliptic_cached(root, cache)
    write_fake(root, seed=1)
    t = cache.stat().st_mtime + 10
    for f in root.iterdir():
        os.utime(f, (t, t))
    b = el.load_elliptic_cached(root, cache)
    assert not np.array_equal(a.X, b.X)
    np.testing.assert_array_equal(b.X, el.load_elliptic(root).X)
    # edges requested but not in the cache -> re-parse
    assert el.load_elliptic_cached(root, cache, with_edges=True).edges is not None
