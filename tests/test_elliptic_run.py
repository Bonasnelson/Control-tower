"""Numerical-only model path and the Elliptic run script, on the synthetic CSVs of test_elliptic.py."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from controltower.data import elliptic as el
from controltower.data.elliptic_adapter import evaluate_labelled, to_tabular
from controltower.data.payments import GLOBAL_ANOMALY, load_yaml
from controltower.models.masked import MaskedFeatureModel
from controltower.scoring import SCORERS, score
from controltower.training.loop import run_regime, score_records
from controltower.training.strategies import make_hooks
from test_elliptic import N_STEPS, PER_STEP, write_fake

ROOT = Path(__file__).resolve().parents[1]


def test_masked_model_numerical_only_trains():
    torch.manual_seed(0)
    m = MaskedFeatureModel([], 6, d_embed=8, hidden=[32], latent=8, dropout=0.0)
    # correlated fields (shared factor + small noise), so a hidden field is predictable from the others
    z = torch.randn(256, 1)
    cat, num = torch.zeros(256, 0, dtype=torch.long), z * torch.linspace(0.5, 1.5, 6) + 0.05 * torch.randn(256, 6)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    mean_loss = lambda: float(np.mean([m.loss(cat, num).item() for _ in range(10)]))  # noqa: E731 (random masks)
    first = mean_loss()
    for _ in range(150):
        opt.zero_grad(); loss = m.loss(cat, num); loss.backward(); opt.step()
    assert mean_loss() < 0.5 * first
    m.eval()
    assert m.field_scores(cat, num).shape == (256, 6) and torch.isfinite(m.score(cat, num)).all()
    assert m.latent(cat, num).shape == (256, 8)


def test_mixed_model_unchanged_by_numerical_only_path():
    torch.manual_seed(0)
    m = MaskedFeatureModel([3, 4], 1, d_embed=4, hidden=[16], latent=4)
    cat, num = torch.tensor([[0, 1], [2, 3]]), torch.rand(2, 1)
    assert m.tokens(cat, num).shape == (2, 3, 4) and m.field_scores(cat, num).shape == (2, 3)


@pytest.fixture
def ell(tmp_path):
    write_fake(tmp_path)
    return el.load_elliptic(tmp_path, with_edges=True)


@pytest.mark.parametrize("with_edges", [False, True])
def test_adapter_and_training(ell, with_edges):
    exps = el.time_step_experiences(ell, 2, 1, N_STEPS)
    data = to_tabular(ell, exps[0], with_edges=with_edges)
    assert data.cat.shape == (len(ell), 0) and data.num.shape[1] == 165 + (95 if with_edges else 0)
    assert np.allclose(data.num[exps[0]].mean(0), 0, atol=1e-4)          # standardised on E1 only
    assert ((data.group == GLOBAL_ANOMALY) == (ell.y == 1)).all()        # labels only in group (eval)
    assert np.abs(data.num).max() <= 10.0
    cfg = load_yaml(ROOT / "configs" / "train.yaml")["masked"] | {"epochs": 1, "buffer_size": 10, "ewc_samples": 16}
    for strategy in ("none", "ewc+replay"):
        model, hist = run_regime("finetune", "masked", data, exps, cfg, seed=0,
                                 hooks=make_hooks(strategy, cfg, data, 0))
        assert len(hist.epoch_loss) == len(exps)
        for name in SCORERS:
            s = score(name, model, data, np.concatenate(exps[:-1]), exps[-1])
            assert s.shape == (len(exps[-1]),) and np.isfinite(s).all()
    assert np.isfinite(score_records(model, data, exps[-1])).all()


def test_evaluate_labelled():
    y = np.array([1, 0, -1, 0, 1, -1, 0, 0, 0, 0])
    s = np.array([9, 1, 8, 2, 7, 0, 3, 0, 0, 0], dtype=float)
    r = evaluate_labelled(s, y, budget=0.2)          # top-2 of all rows: rows 0 and 2
    assert r["auprc"] == pytest.approx(1.0) and r["recall_at_1pct"] == pytest.approx(0.5)
    assert (r["n_illicit"], r["n_licit"], r["n_unknown"]) == (2, 6, 2)
    assert np.isnan(evaluate_labelled(s, np.zeros(10, int))["auprc"])


def test_run_elliptic_script_end_to_end(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    write_fake(raw)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_elliptic.py"), "--root", str(raw),
                        "--cache", str(tmp_path / "cache.npz"), "--out-dir", str(out), "--steps-per-experience", "2",
                        "--epochs", "1", "--seeds", "0", "1", "--scorer", "recon_error", "--threads", "1"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    df = pd.read_csv(out / "elliptic_none_recon_error.csv")
    n_exp = N_STEPS // 2
    assert len(df) == 2 * (n_exp - 1)                                  # 2 seeds x (experiences - 1)
    assert list(df["evaluated"].unique()) == [f"E{k}" for k in range(2, n_exp + 1)]
    assert (df["n"] == 2 * PER_STEP).all() and df["auprc"].between(0, 1).all()
    assert (out / "elliptic_none_recon_error_summary.md").exists()
    # missing data: clear message, non-zero exit, no output
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_elliptic.py"), "--root", str(tmp_path / "none"),
                        "--cache", str(tmp_path / "none.npz"), "--out-dir", str(tmp_path / "out2")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 1 and "Kaggle" in r.stdout and not (tmp_path / "out2").exists()
