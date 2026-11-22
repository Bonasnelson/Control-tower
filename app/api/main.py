"""Read-only API for the Control Tower demo dashboard.

Every endpoint reads files that the project scripts wrote (configs/ and results/);
nothing is computed or invented here. Missing outputs return ``{"status": "pending"}``
with the command that produces them.

    uvicorn main:app --app-dir app/api --port 8000
"""
from __future__ import annotations

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

# Hemati et al. (2022), Table 2, Philadelphia, as quoted in README.md (Week 1 result).
PAPER_REFERENCE = {
    "source": "Hemati et al. (2022), Table 2, Philadelphia (as quoted in README.md)",
    "SEL": {"delta_fp_x1e3": "+1.45 / +1.40 / +1.45", "L_rec_target_x1e3": 6.59},
    "JEL": {"delta_fp_x1e3": "-1.36 ± 0.05", "L_rec_target_x1e3": 2.04},
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
            "rows": rows, "paper": PAPER_REFERENCE,
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
