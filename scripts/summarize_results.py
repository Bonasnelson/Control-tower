"""Merge baseline run files into one mean ± std table (Markdown + LaTeX for the report).

    python scripts/summarize_results.py --city philadelphia --model baseline_ae
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLS = [("delta_fp", r"$\Delta$FP ($\times10^{3}$)", 1e3), ("fp_ratio", "FP ratio", 1),
        ("fn_ratio", "FN ratio", 1), ("auprc", "AUPRC", 1), ("recall_at_1pct", "Recall@1\\%", 1)]
NAMES = {"single": "SEL", "joint": "JEL", "finetune": "SFT"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="philadelphia")
    ap.add_argument("--model", default="baseline_ae")
    a = ap.parse_args()
    files = [f for f in (ROOT / "results").glob(f"baselines_{a.city}_{a.model}_*.csv")
             if "quick" not in f.name and "m1check" not in f.name]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(["regime", "scenario", "seed"])
    g = df.groupby(["regime", "scenario"])
    n = g.size()

    md = [f"# Reproduced baselines: {a.city}, {a.model} (last experience; mean ± std over seeds)", "",
          "| regime | scenario | seeds | " + " | ".join(c[1].replace('$', '').replace('\\', '') for c in COLS) + " |",
          "|---|---|---|" + "---|" * len(COLS)]
    tex = []
    for (reg, scen), sub in g:
        cells_md, cells_tex = [], []
        for col, _, k in COLS:
            m, s = sub[col].mean() * k, sub[col].std(ddof=1) * k if len(sub) > 1 else 0.0
            cells_md.append(f"{m:.3f} ± {s:.3f}")
            cells_tex.append(f"${m:.2f} \\pm {s:.2f}$")
        md.append(f"| {NAMES.get(reg, reg)} | {scen} | {n[(reg, scen)]} | " + " | ".join(cells_md) + " |")
        tex.append(f"{NAMES.get(reg, reg)} & {scen} & " + " & ".join(cells_tex) + r" \\")
    out = ROOT / "results" / f"summary_{a.city}_{a.model}"
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    out.with_suffix(".tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("\n".join(tex))
