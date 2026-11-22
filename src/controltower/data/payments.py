"""City-payments data pipeline (Philadelphia and Chicago).

Reproduces the benchmark construction of Hemati, Schreyer and Borth (2022),
"Continual learning for unsupervised anomaly detection in continuous auditing of
financial accounting data" (arXiv:2112.13215), following the authors' reference
code (github.com/GitiHubi/deepContinualAuditing, ``BenchmarkHandler``):

1. keep the ``tau = 10`` departments with the most payments;
2. inject synthetic *global* anomalies (rare attribute values, group 1000) and
   *local* anomalies (unusual combinations of common values, group 2000);
3. encode categorical attributes as integer codes (one-hot is built on the fly
   by the baseline model) and min-max scale numerical attributes;
4. split into ``N = 10`` experiences: every department contributes a disjoint
   slice of ``n_dept_samples_per_exp`` records per experience, scaled by a
   percentage (U[0.9, 1] for normal departments, a decay schedule for the
   target department A'); anomalies appear only in the last experience.

The data are stored as integer codes rather than one-hot vectors so that
~180k x 5.6k inputs never need to be materialised.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

GLOBAL_ANOMALY = 1000
LOCAL_ANOMALY = 2000
OTHER = "__OTHER__"


@dataclass
class TabularData:
    """Encoded table. ``cat``: (n, F) int64 codes; ``num``: (n, K) float32 in [0, 1]."""

    cat: np.ndarray
    num: np.ndarray
    group: np.ndarray            # department index 0..tau-1, or GLOBAL_/LOCAL_ANOMALY
    cat_names: list[str]
    num_names: list[str]
    cardinalities: list[int]
    vocab: dict[str, list[str]] = field(default_factory=dict)
    dept_names: list[str] = field(default_factory=list)
    target_group: int = 0

    def __len__(self) -> int:
        return len(self.group)

    @property
    def onehot_dim(self) -> int:
        return int(sum(self.cardinalities)) + self.num.shape[1]

    @property
    def is_anomaly(self) -> np.ndarray:
        return self.group >= GLOBAL_ANOMALY

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, cat=self.cat, num=self.num, group=self.group,
            meta=np.array(yaml.safe_dump({
                "cat_names": self.cat_names, "num_names": self.num_names,
                "cardinalities": self.cardinalities, "vocab": self.vocab,
                "dept_names": self.dept_names, "target_group": self.target_group,
            }, allow_unicode=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TabularData":
        z = np.load(path, allow_pickle=False)
        meta = yaml.safe_load(str(z["meta"]))
        return cls(cat=z["cat"], num=z["num"], group=z["group"], **meta)


# --------------------------------------------------------------------------- raw loading
def load_raw(cfg: dict, root: str | Path = ".") -> pd.DataFrame:
    """Read the raw CSV for a city and return a frame with a ``dept_key`` column
    plus the configured categorical / numerical columns (all categoricals as str)."""
    df = pd.read_csv(Path(root) / cfg["raw_file"], low_memory=False, dtype=str)
    city = cfg["city"]
    if city == "philadelphia":
        df["dept_key"] = df["dept"]
    elif city == "chicago":
        df = df[df["department_name"].notna()].copy()
        df["dept_key"] = df["department_name"]
        df["voucher_prefix"] = df["voucher_number"].fillna("NONE").str[:4]
        date = df["check_date"].fillna("")
        # check_date is either "YYYY" (older records) or "MM/DD/YYYY"
        full = date.str.len() == 10
        df["check_year"] = np.where(full, date.str[-4:], date.str[:4])
        df["check_month"] = np.where(full, date.str[:2], "NA")
    else:
        raise ValueError(f"unknown city {city!r}")
    for c in cfg["categorical"]:
        df[c] = df[c].fillna("NA").astype(str).str.strip()
    for c in cfg["numerical"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def select_top_departments(df: pd.DataFrame, n: int) -> tuple[pd.DataFrame, list[str]]:
    counts = df["dept_key"].value_counts()
    keep = list(counts.index[:n])
    return df[df["dept_key"].isin(keep)].reset_index(drop=True), keep


# --------------------------------------------------------------------------- anomalies
def inject_anomalies(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Append synthetic anomalies, marked in a ``group_key`` column.

    *Global* anomalies (group 1000): a real record whose vendor and one more
    categorical field get values never seen in the data, and whose amount is
    pushed far into the upper tail. *Local* anomalies (group 2000): every
    attribute value is common, but each field is copied from a record of a
    different department, giving combinations that do not occur in the data.
    The paper generates both kinds with the Faker library; we use the same
    definitions without the extra dependency.
    """
    cats, nums = cfg["categorical"], cfg["numerical"]
    dept_col = "dept" if cfg["city"] == "philadelphia" else "department_name"
    df = df.copy()
    df["group_key"] = df["dept_key"]

    rows = []
    # global anomalies: rare / unseen values and extreme amounts
    tmpl = df.iloc[rng.integers(0, len(df), cfg["n_global_anomalies"])]
    other_fields = [c for c in cats if c not in (dept_col, "vendor_name")]
    for i, (_, r) in enumerate(tmpl.iterrows()):
        r = r.copy()
        r["vendor_name"] = f"GLOBAL ANOMALY VENDOR {i:04d}"
        f = other_fields[rng.integers(len(other_fields))]
        r[f] = f"GLOBAL_RARE_{f}_{i:04d}"
        for n in nums:
            hi = df[n].quantile(0.999)
            r[n] = float(hi * rng.uniform(2.0, 5.0))
        r["group_key"] = GLOBAL_ANOMALY
        rows.append(r)

    # local anomalies: each field sampled from a randomly chosen *different* department
    by_dept = {d: g for d, g in df.groupby("dept_key")}
    depts = list(by_dept)
    for i in range(cfg["n_local_anomalies"]):
        base_dept = depts[rng.integers(len(depts))]
        r = by_dept[base_dept].iloc[rng.integers(len(by_dept[base_dept]))].copy()
        for c in cats:
            if c == dept_col:
                continue
            other = [d for d in depts if d != base_dept]
            src = by_dept[other[rng.integers(len(other))]]
            r[c] = src[c].iloc[rng.integers(len(src))]
        r["group_key"] = LOCAL_ANOMALY
        rows.append(r)

    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


# --------------------------------------------------------------------------- encoding
def encode(df: pd.DataFrame, cfg: dict, dept_names: list[str]) -> TabularData:
    cats, nums = cfg["categorical"], cfg["numerical"]
    max_vocab = cfg.get("max_vocab")
    cat_codes, cards, vocab = [], [], {}
    for c in cats:
        col = df[c].astype(str)
        counts = col.value_counts()
        if max_vocab and len(counts) > max_vocab:
            keep = set(counts.index[: max_vocab - 1])
            # anomaly-specific values are always kept so they stay "rare", not merged
            keep |= set(col[df["group_key"].isin([GLOBAL_ANOMALY, LOCAL_ANOMALY])])
            col = col.where(col.isin(keep), OTHER)
        values = sorted(col.unique())
        index = {v: i for i, v in enumerate(values)}
        cat_codes.append(col.map(index).to_numpy(np.int64))
        cards.append(len(values))
        vocab[c] = values

    num = df[nums].to_numpy(np.float64)
    lo, hi = num.min(axis=0), num.max(axis=0)
    num = (num - lo) / np.where(hi > lo, hi - lo, 1.0)  # min-max, as in the paper

    dept_index = {d: i for i, d in enumerate(dept_names)}
    group = df["group_key"].map(lambda g: g if g in (GLOBAL_ANOMALY, LOCAL_ANOMALY) else dept_index[g])
    return TabularData(
        cat=np.stack(cat_codes, axis=1), num=num.astype(np.float32),
        group=group.to_numpy(np.int64), cat_names=list(cats), num_names=list(nums),
        cardinalities=cards, vocab=vocab, dept_names=list(dept_names),
        target_group=dept_index[str(cfg["target_dept"])],
    )


def prepare_city(cfg: dict, root: str | Path = ".", seed: int = 0) -> TabularData:
    """Raw CSV -> top departments -> anomalies -> encoded ``TabularData``."""
    rng = np.random.default_rng(seed)
    df = load_raw(cfg, root)
    df, dept_names = select_top_departments(df, cfg["n_departments"])
    df = inject_anomalies(df, cfg, rng)
    return encode(df, cfg, dept_names)


def load_or_prepare(city: str, root: str | Path = ".", seed: int = 0) -> TabularData:
    root = Path(root)
    cache = root / "data" / "processed" / f"{city}_seed{seed}.npz"
    if cache.exists():
        return TabularData.load(cache)
    cfg = load_yaml(root / "configs" / f"{city}.yaml")
    data = prepare_city(cfg, root, seed)
    data.save(cache)
    return data


# --------------------------------------------------------------------------- experiences
def build_experiences(data: TabularData, scenario: dict, rng: np.random.Generator) -> list[np.ndarray]:
    """Split ``data`` into ``n_experiences`` index arrays (``BenchmarkHandler`` logic).

    For each department the shuffled record indices are cut into disjoint slices of
    ``n_dept_samples_per_exp``; experience ``e`` keeps the first ``perc[e]`` share of
    slice ``e``. Departments with fewer than 10,000 records get shorter late slices,
    exactly as in the reference code. Anomaly pools enter only the last experience.
    """
    n_exp = scenario["n_experiences"]
    per = scenario["n_dept_samples_per_exp"]
    n_dept = len(data.dept_names)
    lo = int(round(scenario["min_sampling_perc"] * 100))

    pools = {}
    for g in list(range(n_dept)) + [GLOBAL_ANOMALY, LOCAL_ANOMALY]:
        idx = np.flatnonzero(data.group == g)
        rng.shuffle(idx)
        pools[g] = idx

    experiences = []
    for e in range(n_exp):
        parts = []
        for g in range(n_dept):
            if g == data.target_group:
                p = scenario["target_dept_data_perc"][e]
            else:
                p = rng.integers(lo, 101) / 100.0
            chunk = pools[g][e * per:(e + 1) * per]
            parts.append(chunk[: int(p * len(chunk))])
        if e == n_exp - 1:
            for g in (GLOBAL_ANOMALY, LOCAL_ANOMALY):
                p = scenario["anomaly_dept_data_perc"][e]
                parts.append(pools[g][: int(p * len(pools[g]))])
        experiences.append(np.sort(np.concatenate(parts)))
    return experiences


def experience_summary(data: TabularData, experiences: list[np.ndarray]) -> pd.DataFrame:
    """Records per (experience, group): handy for checking a benchmark."""
    labels = data.dept_names + ["GLOBAL_ANOM", "LOCAL_ANOM"]
    keys = list(range(len(data.dept_names))) + [GLOBAL_ANOMALY, LOCAL_ANOMALY]
    rows = [[int((data.group[ix] == k).sum()) for k in keys] for ix in experiences]
    out = pd.DataFrame(rows, columns=labels)
    out.index = [f"E{i + 1}" for i in range(len(experiences))]
    out["total"] = out.sum(axis=1)
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
