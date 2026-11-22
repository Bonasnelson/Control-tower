"""Corrupted-field recovery and chance baseline of scripts/reason_code_hits.py (synthetic data)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from controltower.data.payments import GLOBAL_ANOMALY, LOCAL_ANOMALY, encode, inject_anomalies, select_top_departments
from test_pipeline import CFG, fake_payments

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reason_code_hits", ROOT / "scripts" / "reason_code_hits.py")
rch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rch)


@pytest.fixture(scope="module")
def data():
    df, depts = select_top_departments(fake_payments(), 4)
    return encode(inject_anomalies(df, CFG, np.random.default_rng(0)), CFG, depts)


def test_global_corrupted_fields(data):
    rows = np.flatnonzero(data.group == GLOBAL_ANOMALY)
    for r in rows:
        bad = rch.corrupted_fields(data, int(r), GLOBAL_ANOMALY, "dept")
        rare = [c for j, c in enumerate(data.cat_names) if data.vocab[c][data.cat[r, j]].startswith("GLOBAL_RARE_")]
        assert len(rare) == 1                      # exactly one extra categorical field is replaced
        assert bad == {"vendor_name", rare[0], *data.num_names}
        vendor = data.vocab["vendor_name"][data.cat[r, data.cat_names.index("vendor_name")]]
        assert vendor.startswith("GLOBAL ANOMALY VENDOR")
        assert "dept" not in bad


def test_local_corrupted_fields(data):
    r = int(np.flatnonzero(data.group == LOCAL_ANOMALY)[0])
    assert rch.corrupted_fields(data, r, LOCAL_ANOMALY, "dept") == {c for c in data.cat_names if c != "dept"}


def test_chance_baseline():
    hit_any, share = rch.chance(8, 3)            # Philadelphia global: 3 of 8 fields corrupted
    assert hit_any == pytest.approx(1 - 10 / 56) and share == pytest.approx(3 / 8)
    assert rch.chance(8, 6) == (1.0, 0.75)        # local: at most 2 clean fields -> a hit is certain
