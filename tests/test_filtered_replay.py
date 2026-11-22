"""Anomaly-aware ("filtered") replay rule and the opt-in linear_spread scenario (synthetic data)."""
from pathlib import Path

import numpy as np
import pytest

from controltower.data.payments import (GLOBAL_ANOMALY, LOCAL_ANOMALY, build_experiences, encode, inject_anomalies,
                                        load_yaml, select_top_departments)
from controltower.training.loop import run_regime
from controltower.training.strategies import ReplayBuffer, anomaly_scores, make_hooks
from test_pipeline import CFG, fake_payments

ROOT = Path(__file__).resolve().parents[1]
TRAIN = load_yaml(ROOT / "configs" / "train.yaml")
SCEN = ROOT / "configs" / "scenarios"


@pytest.fixture(scope="module")
def data():
    df, depts = select_top_departments(fake_payments(), 4)
    return encode(inject_anomalies(df, CFG, np.random.default_rng(0)), CFG, depts)


def scen(name):
    return load_yaml(SCEN / f"{name}.yaml") | {"n_dept_samples_per_exp": 50}


@pytest.fixture(scope="module")
def spread(data):
    return build_experiences(data, scen("linear_spread"), np.random.default_rng(0))


@pytest.fixture(scope="module")
def trained(data, spread):
    model, _ = run_regime("finetune", "masked", data, spread[:3], TRAIN["masked"] | {"epochs": 2}, seed=0)
    return model


# ------------------------------------------------------------------ scenario
def test_linear_spread_scenario(data, spread):
    base = build_experiences(data, scen("linear"), np.random.default_rng(0))
    assert np.array_equal(base[-1], spread[-1])                       # E10 identical to "linear"
    for b, s in zip(base, spread):                                    # normal records identical everywhere
        assert np.array_equal(b[data.group[b] < GLOBAL_ANOMALY], s[data.group[s] < GLOBAL_ANOMALY])
    counts = [(int((data.group[e] == GLOBAL_ANOMALY).sum()), int((data.group[e] == LOCAL_ANOMALY).sum())) for e in spread]
    assert counts == [(0, 0)] * 4 + [(2, 2)] * 6                      # 10% of the 20-record pools, E5..E10
    flat = np.concatenate(spread)
    assert len(flat) == len(np.unique(flat))                          # disjoint


def test_pool_too_small_raises(data):
    s = scen("linear_spread") | {"anomaly_dept_data_perc": [0.0] * 4 + [0.2] * 6}   # needs 120% of the pool
    with pytest.raises(ValueError, match="anomaly pool"):
        build_experiences(data, s, np.random.default_rng(0))


def test_default_scenarios_unchanged_by_spread_support(data):
    # paper scenarios: anomalies only in the last experience, first slice of each pool
    for name in ("linear", "exponential", "instant", "full"):
        exps = build_experiences(data, scen(name), np.random.default_rng(1))
        assert all(not (data.group[e] >= GLOBAL_ANOMALY).any() for e in exps[:-1])
        assert (data.group[exps[-1]] >= GLOBAL_ANOMALY).sum() == 4


# --------------------------------------------------------------------- rule
def test_filtered_drops_top_q_and_respects_size(data, spread, trained):
    rb = ReplayBuffer(data, size=60, rule="filtered", seed=0, filter_q=0.1)
    seen = []
    for e in spread[:6]:
        pool = np.concatenate([rb.buffer, e])
        s = anomaly_scores(trained, data, pool)
        rb.after_experience(trained, e)
        seen.append(e)
        k = int(np.ceil(0.1 * len(pool)))
        assert len(rb.last_dropped) == k
        assert set(rb.last_dropped) == set(pool[np.argsort(-s, kind="stable")[:k]])   # the top-q scores
        assert not np.isin(rb.buffer, rb.last_dropped).any()          # dropped records are not kept
        assert len(rb.buffer) == min(60, len(np.concatenate(seen))) or len(rb.buffer) <= 60
        assert len(np.unique(rb.buffer)) == len(rb.buffer) and np.isin(rb.buffer, np.concatenate(seen)).all()


def test_filtered_q0_equals_random(data, spread, trained):
    a = ReplayBuffer(data, 40, "random", seed=3)
    b = ReplayBuffer(data, 40, "filtered", seed=3, filter_q=0.0)
    for e in spread[:5]:
        a.after_experience(trained, e)
        b.after_experience(trained, e)
        assert np.array_equal(a.buffer, b.buffer) and len(b.last_dropped) == 0


def test_filtered_keeps_injected_out(data, spread, trained):
    # anomalies arrive from E5; with a generous q the filter should remove most injected records it sees
    rb = ReplayBuffer(data, size=200, rule="filtered", seed=0, filter_q=0.3)
    dropped_inj = 0
    for e in spread[:9]:
        rb.after_experience(trained, e)
        dropped_inj += int((data.group[rb.last_dropped] >= GLOBAL_ANOMALY).sum())
    assert dropped_inj > 0


def test_filtered_via_make_hooks_and_finetune(data, spread):
    cfg = TRAIN["masked"] | {"epochs": 1, "buffer_size": 40, "buffer_rule": "filtered", "buffer_filter_q": 0.05}
    model, hist = run_regime("finetune", "masked", data, spread, cfg, seed=0, hooks=make_hooks("replay", cfg, data, 0))
    assert len(hist.epoch_loss) == len(spread) and all(np.isfinite(l).all() for l in hist.epoch_loss)
    with pytest.raises(ValueError):
        ReplayBuffer(data, 10, "filtered", filter_q=1.0)
