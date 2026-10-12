from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import DELAY_THRESHOLD_H

STAGE_NAMES = {0: "Origin departure", 1: "Ocean transit", 2: "Port & customs", 3: "Inland delivery"}
GROUP_LABELS = {"origin": "origin delay/congestion", "weather": "weather", "transit_obs": "observed transit delay",
                "congestion": "port congestion", "port_obs": "observed port/customs behaviour",
                "calendar": "weekend/holiday timing", "carried": "carried delay & buffers (propagation)",
                "route_static": "route/port characteristics", "data_quality": "missing-data signal"}


def stage_attribution(pred, df: pd.DataFrame) -> pd.DataFrame:
    g, base, d = pred.shap_groups(df)
    cp = pred.cp

    def grp(*names):
        return sum((g[n] if n in g else 0.0) for n in names)

    A = pd.DataFrame(index=df.index)
    A["S0"] = d["L0"] + grp("origin")
    A["S1"] = ((d["L1"] - d["L0"]) if cp >= 1 else 0.0) + grp("weather", "transit_obs")
    A["S2"] = ((d["L2"] - d["L1"]) if cp >= 2 else 0.0) + grp("congestion", "port_obs", "calendar")
    A["S3"] = 0.0
    A["other"] = grp("carried", "route_static", "data_quality") + base
    return A


def validate_attribution(attr_oof: pd.DataFrame, prefix: str = "S") -> pd.DataFrame:
    """attr_oof: columns cp, <prefix>0..3, true_phi_stage0..3, true_L3 (out-of-fold). Spearman per stage per checkpoint.
    prefix "S" = GBM/SHAP stage attribution, "B" = chain-model stage breakdown."""
    rows = []
    for cp, g in attr_oof.groupby("cp"):
        for k in range(4):
            a, t = g[f"{prefix}{k}"].to_numpy(), g[f"true_phi_stage{k}"].to_numpy()
            rho = spearmanr(a, t).statistic if np.std(a) > 1e-9 else np.nan
            rows.append(dict(cp=cp, stage=STAGE_NAMES[k], spearman=rho))
        dl = g[g["true_L3"] > DELAY_THRESHOLD_H]
        pa = dl[[f"{prefix}{k}" for k in range(4)]].to_numpy().argmax(1)
        pt = dl[[f"true_phi_stage{k}" for k in range(4)]].to_numpy().argmax(1)
        rows.append(dict(cp=cp, stage="top-stage agreement (delayed shipments)", spearman=float((pa == pt).mean())))
    return pd.DataFrame(rows)


def stage_breakdown(out: pd.Series, cp: int) -> pd.DataFrame:
    """Per-stage change in lateness, from the chain model's milestone lateness (chain_L0..chain_L3).
    delta_k = chain_L_k - chain_L_(k-1) with chain_L_(-1) = 0, so the deltas sum to the chain's mean final delay.
    Negative = schedule buffer recovered time.  status: observed | estimated (a missing timestamp was imputed) |
    expected (stage not reached yet; mean of the simulation)."""
    L = [float(out[f"chain_L{k}"]) for k in range(4)]
    est1, est2 = bool(out.get("L1_imputed", 0)) and cp >= 1, bool(out.get("L2_imputed", 0)) and cp >= 2
    rows, prev = [], 0.0
    for k in range(4):
        if k > cp:
            status = "expected"
        elif (k == 1 and est1) or (k == 2 and (est1 or est2)):
            status = "estimated"
        else:
            status = "observed"
        rows.append(dict(stage=STAGE_NAMES[k], delta_h=L[k] - prev, status=status))
        prev = L[k]
    return pd.DataFrame(rows)


def explain_text(out: pd.Series, bd: pd.DataFrame, cp: int, threshold_h: float, counterfactual: dict | None = None) -> list:
    """Plain-language bullets that only restate model outputs."""
    med, lo, hi = out["delay_chain_median"], out["c10"], out["c90"]
    when = lambda h: f"{abs(h):.1f} h {'late' if h > 0 else 'early'}"
    lines = [f"The median estimate is **{when(med)}** against the scheduled delivery; 80% of simulated outcomes "
             f"fall between {lo:+.1f} h and {hi:+.1f} h."]
    obs = bd[bd.status != "expected"]
    if cp == 0:
        lines.append(f"So far: the shipment departed {when(float(out['L_k']))}.")
    else:
        parts = ", ".join(f"{r.stage.lower()} {r.delta_h:+.1f} h" for r in obs.itertuples())
        lines.append(f"So far: {float(out['L_k']):+.1f} h accumulated ({parts}).")
        if (obs.status == "estimated").any():
            lines.append("A milestone timestamp was missing, so that part is estimated by the model.")
    fut = bd[bd.status == "expected"]
    if len(fut):
        top = fut.sort_values("delta_h", ascending=False).iloc[0]
        if top.delta_h > 0.5:
            lines.append(f"Most of the delay still to come is expected in **{top.stage.lower()}** "
                         f"({top.delta_h:+.1f} h on average).")
        else:
            lines.append("Little additional delay is expected in the remaining stages.")
    rec = obs[(obs.delta_h < -1.0) & (obs.stage != STAGE_NAMES[0])]
    for r in rec.itertuples():
        lines.append(f"Schedule buffer recovered about {abs(r.delta_h):.1f} h during {r.stage.lower()}.")
    if counterfactual:
        lines.append(f"What-if: {counterfactual['label']} would move the median by {counterfactual['delta_h']:+.1f} h.")
    lines.append(f"Estimated probability of arriving more than {threshold_h:g} h late: **{out['p_delay_chain']:.0%}** "
                 "(share of simulated outcomes).")
    return lines
