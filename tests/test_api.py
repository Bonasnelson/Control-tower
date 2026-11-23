"""Dashboard API smoke tests (skipped when fastapi is not installed: see app/requirements.txt)."""
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_app(monkeypatch, root: Path):
    monkeypatch.setenv("CONTROLTOWER_ROOT", str(root))
    spec = importlib.util.spec_from_file_location("ct_api_main", ROOT / "app" / "api" / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return TestClient(mod.app)


def test_real_repo_endpoints(monkeypatch):
    c = load_app(monkeypatch, ROOT)
    assert c.get("/api/health").json()["status"] == "ok"
    cities = c.get("/api/cities").json()
    assert {x["city"] for x in cities["cities"]} >= {"philadelphia", "chicago"}
    assert "linear" in {s["name"] for s in cities["scenarios"]}
    for path in ("/api/experiences/philadelphia", "/api/baselines", "/api/v1/scoring",
                 "/api/v1/alerts/philadelphia", "/api/v1/record/philadelphia/0"):
        r = c.get(path)
        assert r.status_code in (200, 404), path
        assert r.json().get("status") in ("ok", "pending", None), path
    assert c.get("/api/experiences/BAD..").status_code in (400, 404)


def test_pending_and_files(monkeypatch, tmp_path):
    (tmp_path / "configs" / "scenarios").mkdir(parents=True)
    (tmp_path / "configs" / "x.yaml").write_text("city: x\ntarget_dept: '1'\n")
    (tmp_path / "results" / "dashboard").mkdir(parents=True)
    c = load_app(monkeypatch, tmp_path)
    assert c.get("/api/v1/scoring").json()["status"] == "pending"
    assert c.get("/api/v1/alerts/x").json()["status"] == "pending"
    assert c.get("/api/experiences/x").json()["status"] == "pending"
    (tmp_path / "results" / "experiences_x_linear.csv").write_text(",1,2,total\nE1,5,6,11\nE2,3,6,9\n")
    e = c.get("/api/experiences/x").json()
    assert e["departments"] == ["1", "2"] and e["experiences"][1] == {"experience": "E2", "1": 3, "2": 6, "total": 9}
    (tmp_path / "results" / "v1_x_scoring.csv").write_text(
        "city,seed,scorer,auprc,recall_at_1pct,delta_fp,fp_ratio,fn_ratio,train_seconds,score_seconds,epochs\n"
        "x,0,field_surprisal,0.5,0.6,1,0,1,10,1,30\nx,1,field_surprisal,0.7,0.8,1,0,1,10,1,30\n")
    s = c.get("/api/v1/scoring").json()
    a = s["cities"]["x"]["scorers"]["field_surprisal"]["auprc"]
    assert a["mean"] == pytest.approx(0.6) and a["n"] == 2
    rec = {"row": 7, "kind": "global_anomaly", "department": None, "reason_codes": ["f"],
           "fields": [{"name": "f", "value": "v", "surprisal": 3.0}]}
    (tmp_path / "results" / "dashboard" / "x_alerts.json").write_text(json.dumps({
        "city": "x", "seed": 0, "scenario": "linear", "model_file": "m.pt", "experience": "E10", "n_eval": 100,
        "budget": 0.01, "n_alerts": 1, "n_injected": {"global": 1, "local": 0}, "target_department": "1",
        "numeric_note": "", "alerts": {"field_surprisal": [{"rank": 1, "row": 7, "score": 3.0,
                                                            "kind": "global_anomaly"}]},
        "records": {"7": rec}}))
    al = c.get("/api/v1/alerts/x?scorer=field_surprisal").json()
    assert al["alerts"][0]["reason_codes"] == ["f"]
    assert c.get("/api/v1/record/x/7").json()["fields"][0]["surprisal"] == 3.0
    assert c.get("/api/v1/record/x/8").status_code == 404
    assert c.get("/api/v1/alerts/x?scorer=nope").status_code == 404


def test_elliptic_states(monkeypatch, tmp_path):
    # real repo: the expected structure comes from the loader; no metrics are reported in any state
    c = load_app(monkeypatch, ROOT)
    e = c.get("/api/elliptic").json()
    assert e["status"] in ("blocked", "data_present", "results_present")
    assert e["expected"]["local_features"] == 93 and e["expected"]["aggregated_features"] == 72
    assert e["expected"]["drift_step"] == 43 and e["tests"]["file"] == "tests/test_elliptic.py"
    assert "test_stream_no_leakage" in e["tests"]["names"]
    assert {r["function"] for r in e["ready"]} == {"load_elliptic_cached", "edge_features", "delayed_label_stream"}
    assert not any(k in e for k in ("auprc", "metrics", "scores"))

    # temp root: blocked -> data present -> results present
    (tmp_path / "results").mkdir()
    c = load_app(monkeypatch, tmp_path)
    e = c.get("/api/elliptic").json()
    assert e["status"] == "blocked" and not any(f["present"] for f in e["files"])
    d = tmp_path / "data" / "elliptic"
    d.mkdir(parents=True)
    (d / "elliptic_txs_features.csv").write_text("1,1")
    assert c.get("/api/elliptic").json()["status"] == "blocked"  # one of three is not enough
    for f in ("elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"):
        (d / f).write_text("x")
    e = c.get("/api/elliptic").json()
    assert e["status"] == "data_present" and all(f["present"] for f in e["files"]) and e["results_files"] == []
    (tmp_path / "results" / "elliptic_stream.csv").write_text("step")
    e = c.get("/api/elliptic").json()
    assert e["status"] == "results_present" and e["results_files"] == ["elliptic_stream.csv"]


def test_paper_reference(monkeypatch, tmp_path):
    # real repo: values come from results/paper_reference_philadelphia.json, flagged until checked against the PDF
    p = load_app(monkeypatch, ROOT).get("/api/baselines").json()["paper"]
    ref = json.loads((ROOT / "results" / "paper_reference_philadelphia.json").read_text(encoding="utf-8"))
    assert p["file"] == "results/paper_reference_philadelphia.json"
    assert p["verified"] is bool(ref["verified"]) is True
    assert p["verified_by"] == "report agent + master, against research paper.pdf (page 6)"
    assert ref["SEL"]["delta_fp_x1e3"]["linear"]["std"] == 0.115 and ref["JEL"]["delta_fp_x1e3"]["std"] == 0.047
    assert (ref["SEL"]["L_rec_target_x1e3"]["std"], ref["JEL"]["L_rec_target_x1e3"]["std"]) == (0.125, 0.042)
    assert "Table 2" in p["source"] and "Table 3" in p["source"] and p["unit_note"].startswith("The paper states no unit")
    assert p["SEL"]["delta_fp_x1e3"] == "+1.45 / +1.40 / +1.45"
    assert p["JEL"]["delta_fp_x1e3"] == "-1.36 ± 0.05"
    assert (p["SEL"]["L_rec_target_x1e3"], p["JEL"]["L_rec_target_x1e3"]) == (6.59, 2.04)

    # temp root: no file -> paper is null; a verified file -> flag and note pass through
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "summary_philadelphia_baseline_ae.md").write_text(
        "| regime | scenario | seeds | DeltaFP (times10^{3}) | FP ratio | FN ratio | AUPRC | Recall@1% |\n"
        "|---|---|---|---|---|---|---|---|\n| JEL | full | 5 | -1.0 ± 0.1 | 0 ± 0 | 1 ± 0 | 0.5 ± 0.1 | 0.6 ± 0.1 |\n",
        encoding="utf-8")
    c = load_app(monkeypatch, tmp_path)
    assert c.get("/api/baselines").json()["paper"] is None
    (tmp_path / "results" / "paper_reference_philadelphia.json").write_text(json.dumps({
        "paper": "P", "city": "philadelphia", "verified": False, "verified_by": "", "verified_note": "", "unit_note": "u",
        "SEL": {"delta_fp_x1e3": {k: {"mean": v, "std": 0.1} for k, v in
                                  (("linear", 1.0), ("exponential", -0.5), ("instant", 2.0))},
                "L_rec_target_x1e3": {"mean": 1.0, "std": 0.1}},
        "JEL": {"delta_fp_x1e3": {"mean": -1.0, "std": 0.1}, "L_rec_target_x1e3": {"mean": 0.5, "std": 0.01}}}),
        encoding="utf-8")
    p = c.get("/api/baselines").json()["paper"]
    assert p["verified"] is False and p["unit_note"] == "u" and p["JEL"]["L_rec_target_x1e3_std"] == 0.01
    assert p["SEL"]["delta_fp_x1e3"] == "+1.00 / -0.50 / +2.00" and p["JEL"]["delta_fp_x1e3"] == "-1.00 ± 0.10"


def test_strategies(monkeypatch, tmp_path):
    # real repo: one row per configuration, labels, seed 0
    s = load_app(monkeypatch, ROOT).get("/api/strategies").json()
    if s["status"] == "ok":
        for city, c in s["cities"].items():
            assert len(c["rows"]) == 6 and c["seeds"][0] == 0
            assert all(r["seeds"] and r["auprc"]["n"] == len(r["seeds"]) for r in c["rows"])
            assert [r["label"] for r in c["rows"]] == ["None (plain fine-tuning)", "EWC", "Replay, random",
                                                       "Replay, diversity", "Replay, high error", "EWC + replay (random)"]
            if c["buffer"]:
                assert {e["experience"] for e in c["buffer"]["runs"][0]["experiences"]} >= {"E9", "E10"}

    # temp root: pending without CSV; shares computed from counts; quick files ignored
    (tmp_path / "results" / "dashboard").mkdir(parents=True)
    c = load_app(monkeypatch, tmp_path)
    assert c.get("/api/strategies").json()["status"] == "pending"
    (tmp_path / "results" / "strategies_x_quick.csv").write_text("city,seed\nx,0\n")
    assert c.get("/api/strategies").json()["status"] == "pending"
    (tmp_path / "results" / "strategies_x.csv").write_text(
        "city,seed,strategy,buffer_rule,auprc,recall_at_1pct,delta_fp,fp_ratio,fn_ratio,train_seconds,epochs,scorer\n"
        "x,0,none,,0.5,0.6,1.5,0,0.1,10,2,field_surprisal\nx,0,replay,high_error,0.7,0.8,-1,0,0.2,12,2,field_surprisal\n"
        "x,1,none,,0.7,0.6,0.5,0,0.3,12,2,field_surprisal\n")
    (tmp_path / "results" / "strategies_x_linear_spread.csv").write_text("city,seed\ny,0\n")  # scenario variant: ignored
    s = c.get("/api/strategies").json()
    rows = s["cities"]["x"]["rows"]
    assert s["status"] == "ok" and set(s["cities"]) == {"x"}
    assert [r["label"] for r in rows] == ["None (plain fine-tuning)", "Replay, high error"]
    assert rows[0]["buffer_rule"] is None and rows[0]["seeds"] == [0, 1] and rows[1]["seeds"] == [0]
    assert rows[0]["auprc"]["mean"] == pytest.approx(0.6) and rows[0]["auprc"]["std"] == pytest.approx(0.1414, abs=1e-4)
    assert rows[1]["delta_fp"] == {"mean": -1.0, "std": 0.0, "n": 1} and s["cities"]["x"]["buffer"] is None
    (tmp_path / "results" / "dashboard" / "x_buffer.json").write_text(json.dumps({
        "city": "x", "buffer_size": 10, "departments": ["a", "b"], "target_department": "a",
        "runs": [{"strategy": "replay", "buffer_rule": "high_error", "seed": 0, "experiences": [
            {"experience": "E9", "size": 10, "counts": {"a": 4, "b": 6}},
            {"experience": "E10", "size": 10, "counts": {"a": 1, "b": 6, "GLOBAL_ANOM": 2, "LOCAL_ANOM": 1}}]}]}))
    b = c.get("/api/strategies").json()["cities"]["x"]["buffer"]
    e9, e10 = b["runs"][0]["experiences"]
    assert b["runs"][0]["seeds"] == [0]
    assert (e9["target_share"]["mean"], e9["injected"]["mean"]) == (0.4, 0)
    assert (e10["target"]["mean"], e10["injected"]["mean"]) == (1, 3)


def test_strategies_linear_spread(monkeypatch, tmp_path):
    # real repo: separate endpoint; the main endpoint never reads the linear_spread files
    c = load_app(monkeypatch, ROOT)
    sp = c.get("/api/strategies/linear_spread").json()
    main_ = c.get("/api/strategies").json()
    if main_["status"] == "ok":
        assert all(v["source"] == f"results/strategies_{k}.csv" and "spread" not in v for k, v in main_["cities"].items())
    if sp["status"] == "ok":
        assert sp["scenario"] == "linear_spread"
        for city, v in sp["cities"].items():
            assert v["source"] == f"results/strategies_{city}_linear_spread.csv"
            labels = [r["label"] for r in v["rows"]]
            assert "Replay, filtered" in labels and "None (plain fine-tuning)" in labels
            none = next(r for r in v["rows"] if r["strategy"] == "none")
            assert none["buffer_injected_used_total"] is None and none["filter_dropped_injected_total"] is None

    # temp root: pending without files; aggregation over seeds; filtered-only column
    (tmp_path / "results").mkdir()
    c = load_app(monkeypatch, tmp_path)
    assert c.get("/api/strategies/linear_spread").json()["status"] == "pending"
    head = ("city,seed,strategy,buffer_rule,auprc,recall_at_1pct,delta_fp,fn_ratio,"
            "buffer_injected_used_total,buffer_injected_after_E9,filter_dropped_injected_total\n")
    (tmp_path / "results" / "strategies_x_linear_spread.csv").write_text(
        head + "x,0,replay,filtered,0.5,0.6,1,0.2,0,0,100\nx,1,replay,filtered,0.7,0.6,1,0.2,2,1,104\n"
        "x,0,replay,random,0.4,0.5,2,0.3,10,2,\n")
    sp = c.get("/api/strategies/linear_spread").json()
    f, r = sp["cities"]["x"]["rows"]
    assert f["seeds"] == [0, 1] and f["auprc"]["mean"] == pytest.approx(0.6)
    assert f["filter_dropped_injected_total"]["mean"] == pytest.approx(102) and f["buffer_injected_used_total"]["mean"] == 1
    assert r["filter_dropped_injected_total"] is None and r["buffer_injected_after_E9"]["mean"] == 2
    assert c.get("/api/strategies").json()["status"] == "pending"   # main endpoint ignores the spread file
