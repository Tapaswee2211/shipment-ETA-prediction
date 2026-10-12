"""
End-to-end evaluation.
  python -m src.pipeline                                  # synthetic data
  python -m src.pipeline --data my.csv --mapping m.json   # your own data (see docs/BRING_YOUR_OWN_DATA.md)
Options: --n --seed --fast --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .adapters import load_dataset
from .config import CP_NAMES, DELAY_THRESHOLD_H, N_SHIPMENTS, OUT_DIR, SEED
from .evaluate import (clf_metrics, interval_metrics, paired_gain, reg_metrics, run_ablation, run_cv,
                       run_missingness_sweep)
from .explain import explain_text, stage_breakdown, validate_attribution
from .features import add_lateness
from .generator import SOURCES, load_or_generate
from .inputs import as_of
from .models import ETAPredictor


def md(df: pd.DataFrame, floatfmt="{:.2f}") -> str:
    d = df.astype(object)
    cols = list(d.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(("n/a" if np.isnan(v) else floatfmt.format(v)) if isinstance(v, (float, np.floating))
                                         else str(v) for v in r) + " |")
    return "\n".join(lines)


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_SHIPMENTS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--fast", action="store_true", help="fewer repeats (quick smoke run)")
    ap.add_argument("--data", help="CSV with your own shipments")
    ap.add_argument("--mapping", help="JSON column mapping for --data")
    ap.add_argument("--out", help="output directory (default outputs/)")
    a = ap.parse_args()
    t0 = time.time()
    out_dir = Path(a.out) if a.out else OUT_DIR
    out_dir.mkdir(exist_ok=True, parents=True)
    reps = 1 if a.fast else 3

    notes = []
    if a.data:
        df, notes = load_dataset(a.data, a.mapping)
        title = f"{a.data} (n={len(df)})"
    else:
        df = load_or_generate(a.n, a.seed, force=True)
        title = f"synthetic (n={len(df)}, seed={a.seed})"
    has_truth = {"true_L3", "true_phi_stage0"}.issubset(df.columns)
    L = add_lateness(df)
    rep = [f"# Results - {title}\n"]
    if notes:
        rep += ["Data notes:\n"] + [f"- {n}" for n in notes] + [""]

    # ---- 0. dataset sanity
    cols = ["L0", "L1_obs", "L2_obs", "L3"]
    desc = pd.DataFrame({"observed": L[cols].notna().sum(), "mean": L[cols].mean(), "std": L[cols].std(),
                         "p90": L[cols].quantile(.9)}).rename_axis("lateness_h").reset_index()
    rep += ["## Dataset",
            f"Delayed (>{DELAY_THRESHOLD_H:.0f}h) rate: **{(L['L3'] > DELAY_THRESHOLD_H).mean():.1%}**; "
            f"missing port-arrival ts: {df.port_arrival_actual_ts.isna().mean():.1%}, "
            f"missing customs ts: {df.customs_cleared_actual_ts.isna().mean():.1%}\n", md(desc), ""]
    if has_truth:
        share = df[[f"true_phi_{s}" for s in SOURCES]].clip(lower=0).mean()
        share.index = SOURCES
        rep += ["Mean (positive) causal contribution by source, hours (synthetic ground truth):\n",
                md(share.rename("hours").rename_axis("source").reset_index()), ""]

    # ---- 1. main CV
    log("[1/5] random CV (full models + attribution)")
    oof = run_cv(df, "random", attribution=True, seed=a.seed, n_repeats=reps)
    reg = reg_metrics(oof)
    rep += [f"## 1. ETA accuracy by checkpoint (random 5-fold x {reps})", "MAE/RMSE in hours of final ETA error.\n"]
    for cp in sorted(reg.cp.unique()):
        rep += [f"**{CP_NAMES[cp]}**\n", md(reg[reg.cp == cp].drop(columns="cp")), ""]
    pg = pd.concat([paired_gain(oof, "delay_carry", "delay_gbm"), paired_gain(oof, "delay_carry", "delay_chain"),
                    paired_gain(oof, "delay_carry", "delay_chain_median"), paired_gain(oof, "delay_sched", "delay_chain_median")])
    rep += ["Paired MAE improvement over baseline (hours, 95% bootstrap CI; >0 = model better):\n", md(pg), ""]
    rep += [f"## 2. Delay classification (P(delay>{DELAY_THRESHOLD_H:g}h))\n", md(clf_metrics(oof)), ""]
    rep += ["## 3. Prediction intervals (nominal 80%)\n", md(interval_metrics(oof)), ""]
    if has_truth:
        rep += ["## 4. Explanation vs generator ground truth (synthetic data only)",
                "Spearman between each explained stage quantity and the simulator's exact Shapley attribution of the final "
                "delay to that stage (across shipments). The simulator's attribution is its own counterfactual, not "
                "real-world causality. Inland (S3) has no observable signal before delivery, so GBM/SHAP attributes 0 (n/a).\n",
                "**4a. Chain-model stage breakdown (the explanation shown in the app headline)**\n",
                md(validate_attribution(oof, "B")), "",
                "**4b. GBM comparator: observed stage delay + SHAP groups (diagnostics only)**\n",
                md(validate_attribution(oof, "S")), ""]
        plots.fig_attribution(oof, out_dir / "fig_attribution_vs_truth.png", cp=1)
    plots.fig_mae_by_checkpoint(reg, out_dir / "fig_mae_by_checkpoint.png")
    plots.fig_reliability(oof, out_dir / "fig_reliability_coverage.png")
    plots.fig_pred_vs_actual(oof, out_dir / "fig_pred_vs_actual.png")
    plots.fig_eta_convergence(oof, df, out_dir / "fig_eta_convergence.png")
    plots.fig_route_mae(oof, df, out_dir / "fig_route_mae.png")

    # ---- 2. generalisation
    log("[2/5] leave-one-route-out")
    r_oof = run_cv(df, "route", full=True, seed=a.seed)
    log("[3/5] forward-in-time")
    t_oof = run_cv(df, "time", full=True, seed=a.seed)
    for nm, o in [("5. Held-out-route generalisation (leave-one-route-out)", r_oof),
                  ("6. Forward-in-time split (expanding window, delivery embargo)", t_oof)]:
        m = reg_metrics(o, ["delay_sched", "delay_carry", "delay_ridge", "delay_gbm", "delay_chain"])
        rep += [f"## {nm}\n", md(m.pivot(index="model", columns="cp", values="MAE").reset_index()), ""]

    # ---- 3. ablations / robustness
    log("[4/5] ablations")
    ab = run_ablation(df, a.seed, n_repeats=1 if a.fast else 2)
    rep += ["## 7. Ablations (MAE, hours; random CV)\n", md(ab.reset_index()), ""]
    log("[5/5] missing-data sweep")
    sw = run_missingness_sweep(df, seed=a.seed)
    rep += ["## 8. Robustness to extra missing events (GBM MAE, hours)\n",
            md(sw.pivot_table(index=["extra_missing", "cp"], columns="imputation", values="MAE").reset_index()
               .astype({"cp": int})), ""]
    plots.fig_missingness(sw, out_dir / "fig_missingness.png")

    # ---- 4. worked example (the shipment is held out of training and only sees checkpoint-1 information)
    cand = df[L["L3"] > 25]
    row = (cand if len(cand) > 5 else df).iloc[[5 if len(cand) > 5 else 0]]
    P = ETAPredictor(1).fit(df.drop(row.index))
    o = P.predict(as_of(row, 1)).iloc[0]
    bd = stage_breakdown(o, 1)
    rep += ["## 9. Worked example (checkpoint 1; shipment held out of training)", "```"]
    rep += [t.replace("**", "") for t in explain_text(o, bd, 1, DELAY_THRESHOLD_H)]
    rep += ["```", md(bd), f"\nActual final delay: {L.loc[row.index[0], 'L3']:+.1f} h"]
    if has_truth:
        rep += ["Simulator ground-truth attribution (hours; the simulator's own counterfactual, not real-world causality): "
                + json.dumps(row[[f"true_phi_{s}" for s in SOURCES]].iloc[0].round(1).to_dict()), ""]
    plots.fig_breakdown(bd, out_dir / "fig_example_breakdown.png",
                        f"{row.shipment_id.iloc[0]} ({row.origin.iloc[0]}→{row.destination.iloc[0]}) @ {CP_NAMES[1]}")

    # ---- 5. machine-readable summary (read by the app's diagnostics panel)
    summary = dict(n=len(df), seed=a.seed, cv_repeats=reps, delay_threshold_h=DELAY_THRESHOLD_H,
                   mae={int(cp): {r.model: round(float(r.MAE), 3) for r in g.itertuples()} for cp, g in reg.groupby("cp")},
                   interval_coverage_80={int(cp): {r.model: round(float(r.coverage_80), 3) for r in g.itertuples()}
                                         for cp, g in interval_metrics(oof).groupby("cp")},
                   brier={int(cp): {r.model: round(float(r.Brier), 4) for r in g.itertuples() if not pd.isna(r.Brier)}
                          for cp, g in clf_metrics(oof).groupby("cp")})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    (out_dir / "results.md").write_text("\n".join(rep), encoding="utf-8")
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)
    log("\n".join(rep))
    log(f"\nDone in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
