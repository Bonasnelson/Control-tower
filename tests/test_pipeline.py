"""Fast tests on synthetic data: run from a fresh clone without downloading anything."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from controltower.data import elliptic
from controltower.data.payments import (GLOBAL_ANOMALY, LOCAL_ANOMALY, TabularData, build_experiences,
                                        encode, inject_anomalies, load_yaml, select_top_departments)
from controltower.metrics import evaluate_last_experience, reason_codes
from controltower.models.autoencoder import OneHotAutoencoder
from controltower.models.masked import MaskedFeatureModel
from controltower.training.loop import run_regime, score_records

ROOT = Path(__file__).resolve().parents[1]
CFG = {"city": "philadelphia", "categorical": ["fm", "dept", "sub_obj", "vendor_name"],
       "numerical": ["transaction_amount"], "target_dept": "d1", "max_vocab": None,
       "n_global_anomalies": 20, "n_local_anomalies": 20}


def fake_payments(n_dept=4, per=600, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dept):
        for _ in range(per):
            rows.append({"dept": f"d{d}", "fm": str(rng.integers(1, 13)),
                         "sub_obj": f"s{d}_{rng.integers(3)}", "vendor_name": f"v{d}_{rng.integers(5)}",
                         "transaction_amount": float(rng.lognormal(5 + d, 0.5))})
    df = pd.DataFrame(rows)
    df["dept_key"] = df["dept"]
    return df


@pytest.fixture(scope="module")
def data() -> TabularData:
    df, depts = select_top_departments(fake_payments(), 4)
    df = inject_anomalies(df, CFG, np.random.default_rng(0))
    return encode(df, CFG, depts)


@pytest.fixture(scope="module")
def scenario() -> dict:
    s = load_yaml(ROOT / "configs" / "scenarios" / "linear.yaml")
    return {**s, "n_dept_samples_per_exp": 50}


def test_encoding(data):
    assert data.cat.shape == (2400 + 40, 4)
    assert (data.cat.max(0) < np.array(data.cardinalities)).all()
    assert 0.0 <= data.num.min() and data.num.max() <= 1.0
    assert (data.group == GLOBAL_ANOMALY).sum() == 20 and (data.group == LOCAL_ANOMALY).sum() == 20


def test_local_anomalies_are_unseen_combinations(data):
    normal = {tuple(r) for r in data.cat[data.group < GLOBAL_ANOMALY]}
    local = [tuple(r) for r in data.cat[data.group == LOCAL_ANOMALY]]
    assert sum(r not in normal for r in local) >= 18


def test_ten_experiences(data, scenario):
    exps = build_experiences(data, scenario, np.random.default_rng(0))
    assert len(exps) == 10
    flat = np.concatenate(exps)
    assert len(flat) == len(np.unique(flat)), "experiences must be disjoint"
    target = [int((data.group[e] == data.target_group).sum()) for e in exps]
    assert target[0] == 50 and target[-1] == 0 and target == sorted(target, reverse=True)
    for e in exps[:-1]:
        assert not (data.group[e] >= GLOBAL_ANOMALY).any()
    assert (data.group[exps[-1]] == LOCAL_ANOMALY).sum() == 2  # 10% of 20


@pytest.mark.parametrize("kind", ["baseline_ae", "masked"])
@pytest.mark.parametrize("regime", ["single", "joint", "finetune"])
def test_regimes_train_and_score(data, scenario, kind, regime):
    cfg = load_yaml(ROOT / "configs" / "train.yaml")[kind] | {"epochs": 2}
    exps = build_experiences(data, scenario, np.random.default_rng(0))
    model, hist = run_regime(regime, kind, data, exps, cfg, seed=0)
    assert len(hist.epoch_loss) == (10 if regime == "finetune" else 1)
    assert all(np.isfinite(l).all() for l in hist.epoch_loss)
    s = score_records(model, data, exps[-1])
    assert s.shape == (len(exps[-1]),) and np.isfinite(s).all()
    res = evaluate_last_experience(s, data, exps[-1])
    assert 0 <= res["auprc"] <= 1 and "delta_fp" in res


def test_masked_model_learns(data):
    torch.manual_seed(0)
    m = MaskedFeatureModel(data.cardinalities, 1, mask_ratio=0.3)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    cat, num = torch.as_tensor(data.cat), torch.as_tensor(data.num)
    first = m.loss(cat, num).item()
    for _ in range(60):
        opt.zero_grad(); l = m.loss(cat, num); l.backward(); opt.step()
    assert l.item() < 0.6 * first
    m.eval()
    fs = m.field_scores(cat[:5], num[:5])
    assert fs.shape == (5, 5)
    assert len(reason_codes(fs.numpy(), data.cat_names + data.num_names)[0]) == 3


def test_onehot_input(data):
    ae = OneHotAutoencoder(data.cardinalities, 1)
    x = ae.to_input(torch.as_tensor(data.cat[:3]), torch.as_tensor(data.num[:3]))
    assert x.shape == (3, data.onehot_dim)
    assert torch.allclose(x[:, :-1].sum(1), torch.full((3,), 4.0))


def test_elliptic_loader(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    feats = np.column_stack([np.arange(100, 100 + n), np.repeat(np.arange(1, 50), 2)[:n] if n <= 98 else None,
                             rng.normal(size=(n, 165))])
    pd.DataFrame(feats).to_csv(tmp_path / "elliptic_txs_features.csv", header=False, index=False)
    cls = rng.choice(["1", "2", "unknown"], n)
    pd.DataFrame({"txId": np.arange(100, 100 + n), "class": cls}).to_csv(
        tmp_path / "elliptic_txs_classes.csv", index=False)
    pd.DataFrame({"txId1": [100, 101], "txId2": [102, 103]}).to_csv(
        tmp_path / "elliptic_txs_edgelist.csv", index=False)
    d = elliptic.load_elliptic(tmp_path, with_edges=True)
    assert d.X.shape == (n, 165) and set(np.unique(d.y)) <= {-1, 0, 1}
    assert ((d.y == 1) == (cls == "1")).all()
    assert d.edges.tolist() == [[0, 2], [1, 3]]
    exps = elliptic.time_step_experiences(d, 5, 1, 30)
    assert len(exps) == 6 and sum(map(len, exps)) == n
    tr, te = elliptic.standardize(d.X[exps[0]], d.X[exps[1]])
    assert np.allclose(tr.mean(0), 0, atol=1e-5)
    with pytest.raises(FileNotFoundError):
        elliptic.load_elliptic(tmp_path / "missing")
