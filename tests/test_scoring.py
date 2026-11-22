"""Scoring functions on the synthetic payments data of test_pipeline.py."""
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from controltower.data.payments import GLOBAL_ANOMALY, encode, inject_anomalies, load_yaml, select_top_departments
from controltower.metrics import reason_codes
from controltower.scoring import SCORERS, frequency_features, score
from controltower.training.loop import run_regime
from test_pipeline import CFG, fake_payments

ROOT = Path(__file__).resolve().parents[1]
TRAIN = load_yaml(ROOT / "configs" / "train.yaml")


@pytest.fixture(scope="module")
def data():
    df, depts = select_top_departments(fake_payments(), 4)
    return encode(inject_anomalies(df, CFG, np.random.default_rng(0)), CFG, depts)


@pytest.fixture(scope="module")
def split(data):
    rng = np.random.default_rng(0)
    normal = rng.permutation(np.flatnonzero(data.group < GLOBAL_ANOMALY))
    train, held = normal[:1800], normal[1800:]
    eval_idx = np.concatenate([held, np.flatnonzero(data.group == GLOBAL_ANOMALY)])
    return train, eval_idx


@pytest.fixture(scope="module")
def models(data, split):
    out = {}
    for kind, epochs in (("masked", 15), ("baseline_ae", 2)):
        cfg = TRAIN[kind] | {"epochs": epochs}
        out[kind], _ = run_regime("single", kind, data, [split[0]], cfg, seed=0)
    return out


@pytest.mark.parametrize("kind", ["masked", "baseline_ae"])
@pytest.mark.parametrize("name", SCORERS)
def test_scorers_finite(data, split, models, kind, name):
    s = score(name, models[kind], data, *split, seed=0)
    assert s.shape == (len(split[1]),) and np.isfinite(s).all()


def test_isolation_forest_model_free_and_detects(data, split):
    s = score("isolation_forest", None, data, *split, seed=0)
    y = data.group[split[1]] == GLOBAL_ANOMALY
    assert average_precision_score(y, s) > 2 * y.mean()
    assert np.array_equal(s, score("isolation_forest", None, data, *split, seed=0))  # seeded


def test_field_surprisal_detects_global(data, split, models):
    s = score("field_surprisal", models["masked"], data, *split)
    y = data.group[split[1]] == GLOBAL_ANOMALY
    assert average_precision_score(y, s) > 2 * y.mean()
    assert s[y].mean() > s[~y].mean()


def test_latent_distance_small_on_reference(data, split, models):
    train, eval_idx = split
    on_train = score("latent_distance", models["masked"], data, train, train[:200], k=1)
    assert np.allclose(on_train, 0, atol=1e-5)  # each record is its own nearest reference
    s = score("latent_distance", models["masked"], data, train, eval_idx, k=10)
    assert score("latent_distance", models["masked"], data, train, train[:200], k=10).mean() <= s.mean()


@pytest.mark.parametrize("name", ["recon_error", "field_surprisal"])
def test_per_field(data, split, models, name):
    n_fields = len(data.cardinalities) + data.num.shape[1]
    f = score(name, models["masked"], data, *split, per_field=True)
    assert f.shape == (len(split[1]), n_fields)
    np.testing.assert_allclose(f.sum(1), score(name, models["masked"], data, *split), rtol=1e-5)
    codes = reason_codes(f, data.cat_names + data.num_names)
    assert len(codes) == len(split[1]) and len(codes[0]) == 3
    with pytest.raises(ValueError):
        score("isolation_forest", None, data, *split, per_field=True)


def test_recon_error_differs_from_surprisal(data, split, models):
    a = score("recon_error", models["masked"], data, *split)
    b = score("field_surprisal", models["masked"], data, *split)
    assert not np.allclose(a, b)


def test_frequency_features(data, split):
    train = split[0]
    F = frequency_features(data, train, train[:5])
    j0 = data.cat[train, 0]
    assert F.shape == (5, len(data.cardinalities) + data.num.shape[1])
    assert F[0, 0] == pytest.approx((j0 == data.cat[train[0], 0]).mean())


def test_bad_inputs(data, split):
    with pytest.raises(ValueError):
        score("bogus", None, data, *split)
    with pytest.raises(ValueError):
        score("field_surprisal", None, data, *split)
