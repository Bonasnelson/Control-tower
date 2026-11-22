"""Forgetting metric (held-out normal records of E1..E9) and its opt-in wiring in the run scripts."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from controltower.data.payments import GLOBAL_ANOMALY, build_experiences, encode, inject_anomalies, load_yaml, select_top_departments
from controltower.forgetting import ForgettingTracker, chain, holdout_split
from controltower.training.loop import run_regime
from test_pipeline import CFG, fake_payments

ROOT = Path(__file__).resolve().parents[1]
TRAIN = load_yaml(ROOT / "configs" / "train.yaml")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def data():
    df, depts = select_top_departments(fake_payments(), 4)
    return encode(inject_anomalies(df, CFG, np.random.default_rng(0)), CFG, depts)


@pytest.fixture(scope="module")
def exps(data):
    scen = load_yaml(ROOT / "configs" / "scenarios" / "linear.yaml") | {"n_dept_samples_per_exp": 50}
    return build_experiences(data, scen, np.random.default_rng(0))


def test_holdout_split(data, exps):
    train, held = holdout_split(data, exps, per_experience=20, seed=0)
    assert len(train) == len(exps) and len(held) == len(exps) - 1
    for k, h in enumerate(held):
        assert len(h) == 20 and set(h) <= set(exps[k])                 # drawn from its own experience
        assert (data.group[h] < GLOBAL_ANOMALY).all()                  # normal records only
        assert not set(h) & set(train[k]) and set(h) | set(train[k]) == set(exps[k])
    assert np.array_equal(train[-1], exps[-1])                          # E10 untouched
    t2, h2 = holdout_split(data, exps, per_experience=20, seed=0)
    assert all(np.array_equal(a, b) for a, b in zip(h2, held))          # fixed for a seed


def test_tracker_matrix_and_summary(data, exps):
    train, held = holdout_split(data, exps, per_experience=20, seed=0)
    tr = ForgettingTracker(data, held)
    calls = []
    run_regime("finetune", "masked", data, train, TRAIN["masked"] | {"epochs": 1}, seed=0,
               on_experience=chain(tr.on_experience, lambda e, m: calls.append(e), None))
    assert calls == list(range(len(exps)))
    s = tr.summary()
    R = np.array(s["matrix"])
    assert R.shape == (len(exps), len(held)) and np.isfinite(R).all()
    assert s["final"] == R[-1].tolist() and s["final_mean"] == pytest.approx(R[-1].mean())
    assert all(f >= 0 for f in s["forgetting"])                        # final minus the best point since seen
    assert s["forgetting"][0] == pytest.approx(R[-1, 0] - R[:, 0].min())
    assert s["holdout_sizes"] == [20] * len(held)


def test_strategies_script_flag(data, exps):
    rs = load_script("run_strategies")
    cfg = TRAIN["masked"] | {"epochs": 1, "buffer_size": 30}
    res_off, _, comp_off = rs.run_one(data, exps, cfg, "replay", "random", 0)
    assert "forgetting" not in res_off and len(comp_off) == len(exps)  # default: unchanged behaviour
    res_on, _, comp_on = rs.run_one(data, exps, cfg, "replay", "random", 0, forgetting=15)
    f = res_on["forgetting"]
    assert len(f["final"]) == len(exps) - 1 and np.isfinite(f["final_mean"]) and len(comp_on) == len(exps)


def test_v1_script_flag(data, exps, monkeypatch):
    v1 = load_script("run_v1")
    monkeypatch.setattr(v1, "load_or_prepare", lambda *a, **k: data)
    monkeypatch.setattr(v1, "build_experiences", lambda *a, **k: exps)
    cfg = TRAIN["masked"] | {"epochs": 1, "buffer_size": 30, "buffer_rule": "random"}
    off = v1.run_city("philadelphia", [0], cfg, "linear", quick=True)
    assert not any(c.startswith("forgetting") for c in off.columns) and len(off) == 4
    log: list = []
    on = v1.run_city("philadelphia", [0], cfg, "linear", quick=True, forgetting=15, forgetting_log=log)
    assert {"forgetting_final_mean", "forgetting_mean"} <= set(on.columns) and len(log) == 1
    assert np.array(log[0]["matrix"]).shape == (len(exps), len(exps) - 1)


def test_flag_default_off():
    for name in ("run_v1", "run_strategies"):
        src = (ROOT / "scripts" / f"{name}.py").read_text(encoding="utf-8")
        assert '"--forgetting", type=int, default=0' in src
