"""EWC and replay strategies on the synthetic payments data of test_pipeline.py."""
from pathlib import Path

import numpy as np
import pytest
import torch

from controltower.data.payments import build_experiences, encode, inject_anomalies, load_yaml, select_top_departments
from controltower.training.loop import build_model, run_regime, score_records, train_experience
from controltower.training.strategies import (BUFFER_RULES, EWC, STRATEGIES, ReplayBuffer, combine_hooks,
                                              k_center_greedy, make_hooks)
from test_pipeline import CFG, fake_payments

ROOT = Path(__file__).resolve().parents[1]
TRAIN = load_yaml(ROOT / "configs" / "train.yaml")


@pytest.fixture(scope="module")
def data():
    df, depts = select_top_departments(fake_payments(), 4)
    return encode(inject_anomalies(df, CFG, np.random.default_rng(0)), CFG, depts)


@pytest.fixture(scope="module")
def exps(data):
    scen = load_yaml(ROOT / "configs" / "scenarios" / "linear.yaml") | {"n_dept_samples_per_exp": 50}
    return build_experiences(data, scen, np.random.default_rng(0))


@pytest.fixture(scope="module")
def trained(data, exps):
    """A masked model after two experiences of fine-tuning."""
    cfg = TRAIN["masked"] | {"epochs": 2}
    model, _ = run_regime("finetune", "masked", data, exps[:2], cfg, seed=0)
    return model


@pytest.mark.parametrize("kind", ["baseline_ae", "masked"])
def test_ewc_penalty_zero_then_positive(data, exps, kind):
    cfg = TRAIN[kind] | {"epochs": 1}
    torch.manual_seed(0)
    model = build_model(kind, data, cfg)
    ewc = EWC(data, lambda_=1.0, n_samples=128)
    assert ewc.extra_loss(model).item() == 0.0
    train_experience(model, data, exps[0], cfg, np.random.default_rng(0), ewc.hooks())
    assert ewc.extra_loss(model).item() == pytest.approx(0.0, abs=1e-12)  # at the anchor
    assert sum(f.sum().item() for f in ewc.fisher.values()) > 0
    train_experience(model, data, exps[1], cfg, np.random.default_rng(1), ewc.hooks())
    f1 = {n: f.clone() for n, f in ewc.fisher.items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)
    pen = ewc.extra_loss(model)
    assert pen.item() > 0 and pen.requires_grad
    ewc.after_experience(model, exps[2])  # Fishers are summed, never shrink
    assert all((ewc.fisher[n] >= f1[n]).all() for n in f1)


@pytest.mark.parametrize("rule", BUFFER_RULES)
def test_buffer_respects_size(data, exps, trained, rule):
    rb = ReplayBuffer(data, size=60, rule=rule, seed=0)
    assert len(rb.replay_indices()) == 0
    seen = []
    for e in exps[:4]:
        rb.after_experience(trained, e)
        seen.append(e)
        buf = rb.replay_indices()
        assert len(buf) == min(60, sum(map(len, seen)))
        assert len(np.unique(buf)) == len(buf)
        assert np.isin(buf, np.concatenate(seen)).all()


def test_rules_differ(data, exps, trained):
    sel = {}
    for rule in BUFFER_RULES:
        rb = ReplayBuffer(data, size=60, rule=rule, seed=0)
        for e in exps[:3]:
            rb.after_experience(trained, e)
        sel[rule] = set(rb.replay_indices().tolist())
    for a in BUFFER_RULES:
        for b in BUFFER_RULES:
            if a < b:
                assert sel[a] != sel[b], (a, b)
    # high_error keeps the top-scoring records of the pool
    rb = ReplayBuffer(data, size=60, rule="high_error", seed=0)
    rb.after_experience(trained, exps[0])
    s = score_records(trained, data, exps[0])
    assert np.allclose(np.sort(score_records(trained, data, rb.replay_indices())), np.sort(s)[-60:])


def test_k_center_spreads_points():
    rng = np.random.default_rng(0)
    z = np.concatenate([rng.normal(0, 0.01, (100, 2)), [[10, 10], [-10, 10], [10, -10]]])
    chosen = k_center_greedy(z, 4, rng)
    assert {100, 101, 102} <= set(chosen.tolist())


def test_combine_hooks(data, exps, trained):
    ewc, rb = EWC(data, 1.0, n_samples=64), ReplayBuffer(data, 30, "random")
    h = combine_hooks(ewc, rb, ReplayBuffer(data, 20, "random", seed=1))
    assert h.extra_loss(trained).item() == 0.0 and len(h.replay_indices()) == 0
    h.after_experience(trained, exps[0])
    assert ewc.fisher is not None and len(h.replay_indices()) == 50
    assert make_hooks("none", {}, data) is None
    with pytest.raises(ValueError):
        make_hooks("bogus", {}, data)


@pytest.mark.parametrize("kind", ["baseline_ae", "masked"])
@pytest.mark.parametrize("strategy,rule", [(s, "random") for s in STRATEGIES if s != "none"]
                         + [("replay", "diversity"), ("replay", "high_error")])
def test_finetune_with_strategy(data, exps, kind, strategy, rule):
    cfg = TRAIN[kind] | {"epochs": 2, "buffer_size": 40, "buffer_rule": rule, "ewc_samples": 128}
    model, hist = run_regime("finetune", kind, data, exps, cfg, seed=0, hooks=make_hooks(strategy, cfg, data, 0))
    assert len(hist.epoch_loss) == 10 and all(np.isfinite(l).all() for l in hist.epoch_loss)
    assert np.isfinite(score_records(model, data, exps[-1])).all()
