"""Read-only API for the Control Tower demo dashboard.

Every endpoint reads files that the project scripts wrote (configs/ and results/);
nothing is computed or invented here. Missing outputs return ``{"status": "pending"}``
with the command that produces them.

    uvicorn main:app --app-dir app/api --port 8000
"""
from __future__ import annotations

import ast
import csv
import json
import math
import os
import re
from pathlib import Path
from statistics import mean, stdev

import yaml
from fastapi import FastAPI, HTTPException, Query

ROOT = Path(os.environ.get("CONTROLTOWER_ROOT", Path(__file__).resolve().parents[2]))
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"
SCORERS = ["recon_error", "field_surprisal", "latent_distance", "isolation_forest"]
V1_METRICS = ["auprc", "recall_at_1pct", "delta_fp", "fp_ratio", "fn_ratio", "score_seconds", "train_seconds"]

PAPER_FILE = "paper_reference_philadelphia.json"
SEL_ORDER = ["linear", "exponential", "instant"]


def paper_reference() -> dict | None:
    """Paper values from results/paper_reference_philadelphia.json (None if the file is missing),
    with display strings added. ``verified`` says whether they were checked against the PDF."""
    path = RESULTS / PAPER_FILE
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    sel, jel = d["SEL"]["delta_fp_x1e3"], d["JEL"]["delta_fp_x1e3"]
    src = d.get("sources", {})
    return {
        "source": f"{d.get('paper', '')}: ΔFP from {src.get('delta_fp', '?')}; L_Rec(A′) from {src.get('L_rec_target', '?')}",
        "file": f"results/{PAPER_FILE}",
        "verified": bool(d.get("verified")), "verified_by": d.get("verified_by", ""),
        "verified_date": d.get("verified_date", ""), "verified_note": d.get("verified_note", ""),
        "unit_note": d.get("unit_note", ""),
        "SEL": {"delta_fp_x1e3": " / ".join(f"{sel[k]['mean']:+.2f}" for k in SEL_ORDER if k in sel),
                "delta_fp_x1e3_by_scenario": sel,
                "L_rec_target_x1e3": d["SEL"]["L_rec_target_x1e3"]["mean"],
                "L_rec_target_x1e3_std": d["SEL"]["L_rec_target_x1e3"]["std"]},
        "JEL": {"delta_fp_x1e3": f"{jel['mean']:.2f} ± {jel['std']:.2f}",
                "delta_fp_x1e3_std": jel["std"],
                "L_rec_target_x1e3": d["JEL"]["L_rec_target_x1e3"]["mean"],
                "L_rec_target_x1e3_std": d["JEL"]["L_rec_target_x1e3"]["std"]},
    }


app = FastAPI(title="Control Tower demo API", version="0.1")


def pending(what: str, command: str) -> dict:
    return {"status": "pending", "missing": what, "command": command}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def mean_std(values: list[float]) -> dict:
    v = [x for x in values if x is not None]
    if not v:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": mean(v), "std": stdev(v) if len(v) > 1 else 0.0, "n": len(v)}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "root": str(ROOT)}


@app.get("/api/cities")
def cities() -> dict:
    out = []
    for p in sorted(CONFIGS.glob("*.yaml")):
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if "city" in cfg:
            out.append({"city": cfg["city"], "n_departments": cfg.get("n_departments"),
                        "target_dept": str(cfg.get("target_dept")), "categorical": cfg.get("categorical", []),
                        "numerical": cfg.get("numerical", [])})
    scenarios = []
    for p in sorted((CONFIGS / "scenarios").glob("*.yaml")):
        s = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        scenarios.append({"name": p.stem, "reduction_type": s.get("reduction_type"),
                          "n_experiences": s.get("n_experiences"),
                          "target_dept_data_perc": s.get("target_dept_data_perc")})
    return {"cities": out, "scenarios": scenarios}


@app.get("/api/experiences/{city}")
def experiences(city: str, scenario: str = "linear") -> dict:
    if not re.fullmatch(r"[a-z_]+", city) or not re.fullmatch(r"[a-z_]+", scenario):
        raise HTTPException(400, "bad city or scenario")
    path = RESULTS / f"experiences_{city}_{scenario}.csv"
    if not path.exists():
        return pending(path.name, "python scripts/prepare_data.py")
    rows = read_csv(path)
    header = list(rows[0].keys()) if rows else []
    depts = [h for h in header[1:] if h not in ("total",)]
    cfg_path = CONFIGS / f"{city}.yaml"
    target = str(yaml.safe_load(cfg_path.read_text(encoding="utf-8")).get("target_dept")) if cfg_path.exists() else None
    return {"status": "ok", "city": city, "scenario": scenario, "source": f"results/{path.name}",
            "departments": depts, "target_dept": target,
            "experiences": [{"experience": r[header[0]], **{d: int(float(r[d])) for d in depts},
                             "total": int(float(r["total"]))} for r in rows]}


def _parse_md_table(text: str) -> list[dict]:
    lines = [l.strip() for l in text.splitlines() if l.strip().startswith("|")]
    if len(lines) < 2:
        return []
    cols = [c.strip() for c in lines[0].strip("|").split("|")]
    return [dict(zip(cols, (c.strip() for c in l.strip("|").split("|")))) for l in lines[2:]]


def _pm(cell: str) -> dict:
    m = re.match(r"\s*([-+]?[\d.eE+-]+)\s*±\s*([\d.eE+-]+)", cell or "")
    return {"mean": float(m.group(1)), "std": float(m.group(2))} if m else {"mean": num(cell), "std": None}


@app.get("/api/baselines")
def baselines() -> dict:
    path = RESULTS / "summary_philadelphia_baseline_ae.md"
    if not path.exists():
        return pending(path.name, "python scripts/run_baselines.py && python scripts/summarize_results.py")
    rows = []
    for r in _parse_md_table(path.read_text(encoding="utf-8")):
        rows.append({"regime": r.get("regime"), "scenario": r.get("scenario"), "seeds": int(r.get("seeds", 0) or 0),
                     **{k: _pm(r.get(c, "")) for k, c in (
                         ("delta_fp_x1e3", "DeltaFP (times10^{3})"), ("fp_ratio", "FP ratio"),
                         ("fn_ratio", "FN ratio"), ("auprc", "AUPRC"), ("recall_at_1pct", "Recall@1%"))}})
    m1 = []
    m1_path = RESULTS / "baselines_philadelphia_masked_m1check.csv"
    if m1_path.exists():
        for r in read_csv(m1_path):
            m1.append({k: (num(r[k]) if k not in ("regime", "scenario", "model") else r[k])
                       for k in ("regime", "scenario", "seed", "model", "epochs", "auprc", "recall_at_1pct",
                                 "delta_fp", "fp_ratio", "fn_ratio", "train_seconds")})
    return {"status": "ok", "city": "philadelphia", "model": "baseline_ae", "source": f"results/{path.name}",
            "rows": rows, "paper": paper_reference(),
            "m1check": {"source": f"results/{m1_path.name}", "rows": m1} if m1 else None}


@app.get("/api/v1/scoring")
def v1_scoring() -> dict:
    files = sorted(RESULTS.glob("v1_*_scoring.csv"))
    if not files:
        return pending("results/v1_<city>_scoring.csv", "python scripts/run_v1.py")
    out = {}
    for p in files:
        rows = read_csv(p)
        city = rows[0]["city"] if rows else p.stem.split("_")[1]
        seeds = sorted({int(r["seed"]) for r in rows})
        per = {}
        for name in SCORERS:
            rs = [r for r in rows if r["scorer"] == name]
            if rs:
                per[name] = {m: mean_std([num(r.get(m)) for r in rs]) for m in V1_METRICS}
        out[city] = {"source": f"results/{p.name}", "seeds": seeds,
                     "epochs": int(float(rows[0]["epochs"])) if rows else None, "scorers": per}
    return {"status": "ok", "scorers": SCORERS, "cities": out}


def _alerts_file(city: str) -> Path:
    if not re.fullmatch(r"[a-z_]+", city):
        raise HTTPException(400, "bad city")
    return RESULTS / "dashboard" / f"{city}_alerts.json"


@app.get("/api/v1/alerts/{city}")
def v1_alerts(city: str, scorer: str = Query("field_surprisal")) -> dict:
    path = _alerts_file(city)
    if not path.exists():
        return pending(f"results/dashboard/{city}_alerts.json", "python scripts/export_dashboard_data.py")
    d = json.loads(path.read_text(encoding="utf-8"))
    if scorer not in d["alerts"]:
        raise HTTPException(404, f"unknown scorer {scorer!r}")
    alerts = []
    for a in d["alerts"][scorer]:
        rec = d["records"].get(str(a["row"]), {})
        alerts.append({**a, "department": rec.get("department"), "reason_codes": rec.get("reason_codes", [])})
    meta = {k: d[k] for k in ("city", "seed", "scenario", "model_file", "experience", "n_eval", "budget",
                              "n_alerts", "n_injected", "target_department", "numeric_note")}
    return {"status": "ok", **meta, "scorer": scorer, "scorers": list(d["alerts"]),
            "source": f"results/dashboard/{path.name}", "alerts": alerts}


@app.get("/api/v1/record/{city}/{row}")
def v1_record(city: str, row: int) -> dict:
    path = _alerts_file(city)
    if not path.exists():
        return pending(f"results/dashboard/{city}_alerts.json", "python scripts/export_dashboard_data.py")
    rec = json.loads(path.read_text(encoding="utf-8"))["records"].get(str(row))
    if rec is None:
        raise HTTPException(404, "record not among the exported alerts")
    return {"status": "ok", "city": city, **rec}


# ------------------------------------------------------------------ Elliptic (blocked until data)
ELLIPTIC_FILES = ["elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"]
ELLIPTIC_READY = [
    ("load_elliptic_cached", "parses the three CSVs once and caches them in data/processed/elliptic.npz"),
    ("edge_features", "in/out degree and the mean local features of each transaction's neighbours"),
    ("delayed_label_stream", "predict-then-learn stream: score step t, then reveal labels of step t - delay"),
]


def _module_info(path: Path) -> dict:
    """Constants and docstring of the loader, read with ast (no import, no data needed)."""
    if not path.exists():
        return {"constants": {}, "doc": ""}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return {"constants": consts, "doc": ast.get_docstring(tree) or ""}


def _test_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]


@app.get("/api/elliptic")
def elliptic() -> dict:
    data_dir = ROOT / "data" / "elliptic"
    files = [{"name": f, "present": (data_dir / f).exists()} for f in ELLIPTIC_FILES]
    info = _module_info(ROOT / "src" / "controltower" / "data" / "elliptic.py")
    c = info["constants"]
    results = sorted(p.name for p in RESULTS.glob("elliptic_*"))
    status = "blocked" if not all(f["present"] for f in files) else ("results_present" if results else "data_present")
    return {
        "status": status,
        "data_dir": "data/elliptic/",
        "files": files,
        "source": {"name": "Kaggle: ellipticco/elliptic-data-set", "url": "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set",
                   "note": "needs a Kaggle login; download and unzip the three CSV files into data/elliptic/"},
        "expected": {  # from the loader's constants and docstring, not measured
            "local_features": c.get("N_LOCAL"), "aggregated_features": c.get("N_AGG"),
            "time_steps": "1..49", "drift_step": c.get("SHUTDOWN_STEP"),
            "labels": {k: v for k, v in (c.get("LABEL_MAP") or {}).items()},
            "from": "src/controltower/data/elliptic.py",
        },
        "ready": [{"function": f, "what": w} for f, w in ELLIPTIC_READY],
        "tests": {"file": "tests/test_elliptic.py", "names": _test_names(ROOT / "tests" / "test_elliptic.py"),
                  "data": "synthetic CSVs the tests write (6 time steps × 20 transactions)"},
        "results_files": results,
    }


# ----------------------------------------------------------------------- strategies (D2/D3)
STRATEGY_METRICS = ["auprc", "recall_at_1pct", "delta_fp", "fp_ratio", "fn_ratio", "train_seconds"]
ANOMALY_GROUPS = ("GLOBAL_ANOM", "LOCAL_ANOM")


def strategy_label(strategy: str, rule: str | None) -> str:
    base = {"none": "None (plain fine-tuning)", "ewc": "EWC", "replay": "Replay", "ewc+replay": "EWC + replay"}.get(strategy, strategy)
    if not rule:
        return base
    rule_txt = rule.replace("_", " ")
    return f"{base}, {rule_txt}" if strategy == "replay" else f"{base} ({rule_txt})"


def _buffer_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    target = d["target_department"]
    runs = []
    for r in d["runs"]:
        exps = []
        for e in r["experiences"]:
            counts = e["counts"]
            size = sum(counts.values()) or 1
            injected = sum(counts.get(k, 0) for k in ANOMALY_GROUPS)
            t = counts.get(target, 0)
            exps.append({"experience": e["experience"], "size": e["size"], "target": t, "injected": injected,
                         "other": size - t - injected, "target_share": t / size, "injected_share": injected / size,
                         "other_share": (size - t - injected) / size})
        runs.append({"strategy": r["strategy"], "buffer_rule": r["buffer_rule"], "seed": r["seed"],
                     "label": strategy_label(r["strategy"], r["buffer_rule"]), "experiences": exps})
    return {"source": f"results/dashboard/{path.name}", "target_department": target,
            "buffer_size": d.get("buffer_size"), "runs": runs}


@app.get("/api/strategies")
def strategies() -> dict:
    files = sorted(RESULTS.glob("strategies_*.csv"))
    files = [p for p in files if not p.stem.endswith("_quick")]
    if not files:
        return pending("results/strategies_<city>.csv", "python scripts/run_strategies.py --seeds 0")
    out = {}
    for p in files:
        rows = read_csv(p)
        if not rows:
            continue
        city = rows[0]["city"]
        out[city] = {
            "source": f"results/{p.name}",
            "seeds": sorted({int(r["seed"]) for r in rows}),
            "epochs": int(float(rows[0]["epochs"])),
            "scorer": rows[0].get("scorer", ""),
            "rows": [{"strategy": r["strategy"], "buffer_rule": r["buffer_rule"] or None,
                      "label": strategy_label(r["strategy"], r["buffer_rule"] or None), "seed": int(r["seed"]),
                      **{m: num(r.get(m)) for m in STRATEGY_METRICS}} for r in rows],
            "buffer": _buffer_summary(RESULTS / "dashboard" / f"{city}_buffer.json"),
        }
    return {"status": "ok", "metrics": STRATEGY_METRICS, "cities": out}
