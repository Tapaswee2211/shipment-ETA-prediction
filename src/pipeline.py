"""End-to-end run:  python -m src.pipeline [--n 300] [--seed 42] [--fast]"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

from .config import CP_NAMES, DELAY_THRESHOLD_H, N_SHIPMENTS, OUT_DIR, SEED
from .evaluate import (clf_metrics, interval_metrics, paired_gain, reg_metrics, run_ablation, run_cv,
                       run_missingness_sweep)
from .explain import narrate, stage_attribution, validate_attribution
from .features import add_lateness
from .generator import SOURCES, load_or_generate
from .models import ETAPredictor
from . import plots

warnings.filterwarnings("ignore")


def md(df: pd.DataFrame, floatfmt="{:.2f}") -> str:
    d = df.astype(object)
    cols = list(d.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(("n/a" if np.isnan(v) else floatfmt.format(v)) if isinstance(v, (float, np.floating))
                                         else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_SHIPMENTS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--fast", action="store_true", help="fewer repeats (quick smoke run)")
    a = ap.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    reps = 1 if a.fast else 3
    df = load_or_generate(a.n, a.seed, force=True)
    L = add_lateness(df)
    rep = [f"# Results  (n={len(df)}, seed={a.seed})\n"]

    # ---- 0. dataset sanity
    desc = pd.DataFrame({"mean": L[["L0", "true_L1", "true_L2", "L3"]].mean(),
                         "std": L[["L0", "true_L1", "true_L2", "L3"]].std(),
                         "p90": L[["L0", "true_L1", "true_L2", "L3"]].quantile(.9)}).reset_index(names="lateness_h")
    stage_share = df[[f"true_phi_{s}" for s in SOURCES]].clip(lower=0).mean()
    stage_share.index = SOURCES
    rep += ["## Dataset", f"Delayed (>{DELAY_THRESHOLD_H:.0f}h) rate: **{(L['L3'] > DELAY_THRESHOLD_H).mean():.1%}**; "
            f"missing port-arrival ts: {df.port_arrival_actual_ts.isna().mean():.1%}, "
            f"missing customs ts: {df.customs_cleared_actual_ts.isna().mean():.1%}\n", md(desc),
            "\nMean (positive) causal contribution by source, hours:\n",
            md(stage_share.rename("hours").rename_axis("source").reset_index()), ""]

    # ---- 1. main CV
    print("[1/5] random CV (full models + attribution)"); oof = run_cv(df, "random", attribution=True, seed=a.seed, n_repeats=reps)
    reg = reg_metrics(oof)
    rep += ["## 1. ETA accuracy by checkpoint (random 5-fold x %d)" % reps, "MAE/RMSE in hours of final ETA error.\n"]
    for cp in sorted(reg.cp.unique()):
        rep += [f"**{CP_NAMES[cp]}**\n", md(reg[reg.cp == cp].drop(columns="cp")), ""]
    pg = pd.concat([paired_gain(oof, "delay_carry", "delay_gbm"), paired_gain(oof, "delay_carry", "delay_chain"),
                    paired_gain(oof, "delay_sched", "delay_chain")])
    rep += ["Paired MAE improvement over baseline (hours, 95% bootstrap CI; >0 = model better):\n", md(pg), ""]
    rep += ["## 2. Delay classification (P(delay>%gh))\n" % DELAY_THRESHOLD_H, md(clf_metrics(oof)), ""]
    rep += ["## 3. Prediction intervals (nominal 80%)\n", md(interval_metrics(oof)), ""]

    # ---- 2. explanations vs ground truth
    attr = validate_attribution(oof)
    rep += ["## 4. Explanation validity vs generator ground truth",
            "Spearman between explained stage contribution and exact Shapley causal contribution (across shipments). "
            "Inland (S3) is n/a: nothing about inland execution is observable before delivery, so the model "
            "(correctly) attributes 0 to it. The last row is top-1 stage agreement, not a correlation.\n",
            md(attr), ""]
    plots.fig_mae_by_checkpoint(reg, OUT_DIR / "fig_mae_by_checkpoint.png")
    plots.fig_reliability(oof, OUT_DIR / "fig_reliability_coverage.png")
    plots.fig_attribution(oof, OUT_DIR / "fig_attribution_vs_truth.png", cp=1)

    # ---- 3. generalisation
    print("[2/5] leave-one-route-out"); r_oof = run_cv(df, "route", full=True, seed=a.seed)
    print("[3/5] forward-in-time"); t_oof = run_cv(df, "time", full=True, seed=a.seed)
    for nm, o in [("5. Held-out-route generalisation (leave-one-route-out)", r_oof),
                  ("6. Forward-in-time split (expanding window, delivery embargo)", t_oof)]:
        m = reg_metrics(o, ["delay_sched", "delay_carry", "delay_ridge", "delay_gbm", "delay_chain"])
        rep += [f"## {nm}\n", md(m.pivot(index="model", columns="cp", values="MAE").reset_index()), ""]

    # ---- 4. ablations / robustness
    print("[4/5] ablations"); ab = run_ablation(df, a.seed, n_repeats=1 if a.fast else 2)
    rep += ["## 7. Ablations (MAE, hours; random CV)\n", md(ab.reset_index()), ""]
    print("[5/5] missing-data sweep"); sw = run_missingness_sweep(df, seed=a.seed)
    rep += ["## 8. Robustness to extra missing events (GBM MAE, hours)\n",
            md(sw.pivot_table(index=["extra_missing", "cp"], columns="imputation", values="MAE").reset_index()
               .astype({"cp": int})), ""]
    plots.fig_missingness(sw, OUT_DIR / "fig_missingness.png")

    # ---- 5. worked example (hold the shipment out)
    row = df[(df.true_L3 > 25)].iloc[[5]]
    P = ETAPredictor(1).fit(df.drop(row.index))
    normal = float(df[df.destination == row.destination.iloc[0]]["dest_congestion_arr"].median())
    story = narrate(P, row, normal_congestion=normal)
    A = stage_attribution(P, row).iloc[0]
    plots.fig_waterfall(A, None, OUT_DIR / "fig_example_waterfall.png",
                        f"{row.shipment_id.iloc[0]} ({row.origin.iloc[0]}→{row.destination.iloc[0]}) @ {CP_NAMES[1]}")
    truth = row[[f"true_phi_{s}" for s in SOURCES]].iloc[0].round(1).to_dict()
    rep += ["## 9. Worked example", "```", story, "```",
            "Ground-truth causal contributions (hours): " + json.dumps(truth), ""]

    (OUT_DIR / "results.md").write_text("\n".join(rep), encoding="utf-8")
    oof.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    print("\n".join(rep))
    print(f"\nDone in {time.time() - t0:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
