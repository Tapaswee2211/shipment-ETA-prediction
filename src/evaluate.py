from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import KFold

from .config import CHECKPOINTS, DELAY_THRESHOLD_H, SEED
from .explain import stage_attribution
from .features import add_lateness
from .models import ETAPredictor

REG_MODELS = ["delay_sched", "delay_carry", "delay_route_prior", "delay_ridge", "delay_gbm", "delay_chain",
              "delay_chain_median"]


def splits(df: pd.DataFrame, kind: str, seed: int = SEED, n_repeats: int = 3):
    n = len(df)
    if kind == "random":
        for rep in range(n_repeats):
            for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=seed + rep).split(df)):
                yield f"r{rep}f{k}", tr, te
    elif kind == "route":
        rid = df["route_id"].to_numpy()
        for r in sorted(set(rid)):
            yield r, np.where(rid != r)[0], np.where(rid == r)[0]
    elif kind == "time":
        order = np.argsort(df["dep_sched_ts"].to_numpy())
        cuts = [0.4, 0.55, 0.7, 0.85, 1.0]
        for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            te = order[int(a * n):int(b * n)]
            t0 = df["dep_sched_ts"].iloc[te].min()
            tr = np.array([j for j in order[:int(a * n)] if df["delivery_actual_ts"].iloc[j] <= t0])  # embargo
            yield f"t{i}", tr, te
    else:
        raise ValueError(kind)


def run_cv(df, kind="random", cps=CHECKPOINTS, full=True, attribution=False, seed=SEED, n_repeats=3, **kw):
    rows = []
    for sid, tr, te in splits(df, kind, seed, n_repeats):
        dtr, dte = df.iloc[tr], df.iloc[te]
        truth = add_lateness(dte)["L3"].to_numpy()
        for cp in cps:
            P = ETAPredictor(cp, seed=seed, full=full, **kw).fit(dtr)
            out = P.predict(dte)
            out["shipment_id"] = dte["shipment_id"].to_numpy()
            out["route_id"] = dte["route_id"].to_numpy()
            out["split"], out["cp"], out["true_L3"] = sid, cp, truth
            out["y"] = (truth > DELAY_THRESHOLD_H).astype(int)
            if attribution:
                A = stage_attribution(P, dte)
                for c in A:
                    out[c] = A[c].to_numpy()
                for k in range(4):                              # chain-model stage breakdown (what the app shows)
                    out[f"B{k}"] = out[f"chain_L{k}"] - (out[f"chain_L{k - 1}"] if k else 0.0)
                for k in range(4):
                    if f"true_phi_stage{k}" in dte:            # ground truth only exists for synthetic data
                        out[f"true_phi_stage{k}"] = dte[f"true_phi_stage{k}"].to_numpy()
            rows.append(out)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- metrics
def _boot_ci(x, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(B, len(x))).mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def reg_metrics(oof, models=REG_MODELS) -> pd.DataFrame:
    rows = []
    for cp, g in oof.groupby("cp"):
        for m in models:
            if m not in g:
                continue
            e = g[m] - g["true_L3"]
            ae = e.abs().groupby(g["shipment_id"]).mean().to_numpy()      # avg over repeats -> per-shipment
            lo, hi = _boot_ci(ae)
            rows.append(dict(cp=cp, model=m.replace("delay_", ""), MAE=e.abs().mean(), MAE_lo=lo, MAE_hi=hi,
                             RMSE=float(np.sqrt((e ** 2).mean())), bias=e.mean()))
    return pd.DataFrame(rows)


def paired_gain(oof, base="delay_carry", model="delay_gbm") -> pd.DataFrame:
    """MAE(base) - MAE(model) with paired bootstrap CI over shipments (positive = model better)."""
    rows = []
    for cp, g in oof.groupby("cp"):
        d = ((g[base] - g["true_L3"]).abs() - (g[model] - g["true_L3"]).abs()).groupby(g["shipment_id"]).mean().to_numpy()
        lo, hi = _boot_ci(d)
        rows.append(dict(cp=cp, base=base.replace("delay_", ""), model=model.replace("delay_", ""),
                         gain_h=d.mean(), lo=lo, hi=hi))
    return pd.DataFrame(rows)


def clf_metrics(oof) -> pd.DataFrame:
    rows = []
    for cp, g in oof.groupby("cp"):
        y = g["y"].to_numpy()
        for name, col, prob in [("gbm_classifier", "p_delay_clf", True), ("chain_monte_carlo", "p_delay_chain", True),
                                ("carry_forward_rule", "delay_carry", False)]:
            if col not in g:
                continue
            s = g[col].to_numpy()
            r = dict(cp=cp, model=name, base_rate=y.mean(), PR_AUC=average_precision_score(y, s),
                     ROC_AUC=roc_auc_score(y, s))
            if prob:
                r["Brier"] = brier_score_loss(y, s)
                r["F1@0.5"] = f1_score(y, s >= 0.5)
            rows.append(r)
    return pd.DataFrame(rows)


def interval_metrics(oof) -> pd.DataFrame:
    rows = []
    for cp, g in oof.groupby("cp"):
        for name, lo, hi in [("gbm_quantile", "p10", "p90"), ("chain_monte_carlo", "c10", "c90")]:
            if lo in g:
                y = g["true_L3"]
                rows.append(dict(cp=cp, model=name, coverage_80=float(((y >= g[lo]) & (y <= g[hi])).mean()),
                                 mean_width_h=float((g[hi] - g[lo]).mean())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- ablations & robustness
ABLATIONS = {
    "full": {},
    "- interactions (queue/congestion x processing)": dict(interactions=False),
    "- propagation features (buffers, increments, risks)": dict(propagation=False),
    "- route priors": dict(priors=False),
    "- missing-data model (LOCF instead)": dict(imputation="locf"),
    "- all four": dict(interactions=False, propagation=False, priors=False, imputation="locf"),
}


def run_ablation(df, seed=SEED, n_repeats=2) -> pd.DataFrame:
    """Feature-group ablations for the boosted model AND the linear model (linear models cannot learn
    interactions by themselves, so engineered features should matter more there)."""
    rows = []
    for name, kw in ABLATIONS.items():
        oof = run_cv(df, "random", full=False, seed=seed, n_repeats=n_repeats, **kw)
        for _, r in reg_metrics(oof, ["delay_gbm", "delay_ridge"]).iterrows():
            rows.append(dict(config=name, model=r["model"], cp=int(r["cp"]), MAE=r["MAE"]))
    t = pd.DataFrame(rows)
    g = t[t.model == "gbm"].pivot(index="config", columns="cp", values="MAE").loc[list(ABLATIONS)]
    g.columns = [f"gbm_cp{c}" for c in g.columns]
    g["gbm_mean"] = g.mean(axis=1)
    rg = t[t.model == "ridge"].pivot(index="config", columns="cp", values="MAE").loc[list(ABLATIONS)]
    g["ridge_mean"] = rg.mean(axis=1)
    return g


def add_missingness(df: pd.DataFrame, q: float, seed: int = 0) -> pd.DataFrame:
    """Additionally hide intermediate events with probability q (MCAR) -> robustness sweep."""
    d = df.copy()
    rng = np.random.default_rng(seed)
    m1, m2 = rng.random(len(d)) < q, rng.random(len(d)) < q
    d.loc[m1, "port_arrival_actual_ts"] = pd.NaT
    d.loc[m2, ["customs_cleared_actual_ts", "port_wait_h", "processing_h", "inspection_flag"]] = np.nan
    return d


def run_missingness_sweep(df, levels=(0.0, 0.1, 0.2, 0.3, 0.45), seed=SEED) -> pd.DataFrame:
    rows = []
    for q in levels:
        dq = add_missingness(df, q, seed)
        for mode in ("model", "locf"):
            oof = run_cv(dq, "random", cps=(1, 2), full=False, seed=seed, n_repeats=1, imputation=mode)
            for _, r in reg_metrics(oof, ["delay_gbm"]).iterrows():
                rows.append(dict(extra_missing=q, imputation=mode, cp=int(r["cp"]), MAE=r["MAE"]))
    return pd.DataFrame(rows)
