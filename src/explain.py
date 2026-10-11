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


def validate_attribution(attr_oof: pd.DataFrame) -> pd.DataFrame:
    """attr_oof: columns cp, S0..S3, true_phi_stage0..3, true_L3 (out-of-fold). Spearman per stage per checkpoint."""
    rows = []
    for cp, g in attr_oof.groupby("cp"):
        for k in range(4):
            a, t = g[f"S{k}"].to_numpy(), g[f"true_phi_stage{k}"].to_numpy()
            rho = spearmanr(a, t).statistic if np.std(a) > 1e-9 else np.nan
            rows.append(dict(cp=cp, stage=STAGE_NAMES[k], spearman=rho))
        dl = g[g["true_L3"] > DELAY_THRESHOLD_H]
        pa = dl[[f"S{k}" for k in range(4)]].to_numpy().argmax(1)
        pt = dl[[f"true_phi_stage{k}" for k in range(4)]].to_numpy().argmax(1)
        rows.append(dict(cp=cp, stage="top-stage agreement (delayed shipments)", spearman=float((pa == pt).mean())))
    return pd.DataFrame(rows)


def what_if(pred, df_row: pd.DataFrame, congestion: float | None = None, weather: float | None = None):
    d2 = df_row.copy()
    if congestion is not None:
        d2["dest_congestion_dep"] = congestion
        d2["dest_congestion_arr"] = congestion
    if weather is not None:
        d2["weather_fcst"] = weather
        d2["weather_actual"] = weather
    return pred.predict(d2)


def narrate(pred, row: pd.DataFrame, normal_congestion: float | None = None) -> str:
    """Human-readable explanation of one shipment's prediction."""
    out = pred.predict(row).iloc[0]
    A = stage_attribution(pred, row).iloc[0]
    g, base, _ = pred.shap_groups(row)
    g = g.iloc[0]
    lines = [f"Predicted final delay: {out['delay_gbm']:+.1f} h"
             + (f"  (P10 {out['p10']:+.1f} / P90 {out['p90']:+.1f} h; P(delay>{DELAY_THRESHOLD_H:.0f}h)="
                f"{out['p_delay_clf']:.0%})" if "p10" in out else "")]
    for k in range(4):
        lines.append(f"  {STAGE_NAMES[k]:<16} {A[f'S{k}']:+6.1f} h")
    lines.append(f"  {'baseline/propagation':<16} {A['other']:+6.1f} h")
    top = g.abs().sort_values(ascending=False).head(3).index
    lines.append("Top model drivers: " + ", ".join(f"{GROUP_LABELS[t]} ({g[t]:+.1f} h)" for t in top))
    if normal_congestion is not None:
        cf = what_if(pred, row, congestion=normal_congestion).iloc[0]["delay_gbm"]
        lines.append(f"Counterfactual: with normal congestion ({normal_congestion:.2f}) delay would be "
                     f"{cf:+.1f} h ({cf - out['delay_gbm']:+.1f} h).")
    return "\n".join(lines)
