import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CP_NAMES, DELAY_THRESHOLD_H
from .explain import STAGE_NAMES

COL = {"sched": "#9ca3af", "carry": "#f59e0b", "route_prior": "#a78bfa", "ridge": "#60a5fa",
       "gbm": "#10b981", "chain": "#fca5a5", "chain_median": "#dc2626"}


def fig_mae_by_checkpoint(reg, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    cps = sorted(reg["cp"].unique())
    names = ["sched", "carry", "route_prior", "ridge", "gbm", "chain", "chain_median"]
    w = 0.11
    for i, m in enumerate(names):
        g = reg[reg.model == m].sort_values("cp")
        if g.empty:
            continue
        x = np.arange(len(cps)) + (i - (len(names) - 1) / 2) * w
        ax.bar(x, g["MAE"], w, color=COL[m], label=m,
               yerr=[g["MAE"] - g["MAE_lo"], g["MAE_hi"] - g["MAE"]], capsize=2, ecolor="#374151")
    ax.set_xticks(range(len(cps)), [CP_NAMES[c] for c in cps])
    ax.set_ylabel("ETA MAE (hours)")
    ax.set_title("ETA error falls as the shipment progresses (95% bootstrap CI)")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_reliability(oof, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    g = oof[oof.cp == 1]
    for col, name, c in [("p_delay_clf", "GBM classifier", "#10b981"), ("p_delay_chain", "Chain Monte-Carlo", "#ef4444")]:
        bins = pd.cut(g[col], np.linspace(0, 1, 6), include_lowest=True)
        t = g.groupby(bins, observed=True).agg(p=(col, "mean"), y=("y", "mean"), n=("y", "size"))
        axes[0].plot(t["p"], t["y"], "o-", color=c, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="predicted P(delay)", ylabel="observed frequency", title="Reliability @ CP1")
    axes[0].legend(frameon=False, fontsize=8)
    for cp in sorted(oof.cp.unique()):
        h = oof[oof.cp == cp]
        axes[1].scatter([cp - .05], [((h.true_L3 >= h.p10) & (h.true_L3 <= h.p90)).mean()], color="#10b981", s=60)
        axes[1].scatter([cp + .05], [((h.true_L3 >= h.c10) & (h.true_L3 <= h.c90)).mean()], color="#ef4444", s=60)
    axes[1].axhline(0.8, color="k", ls="--", lw=1)
    axes[1].set(ylim=(0.4, 1), xticks=sorted(oof.cp.unique()), xlabel="checkpoint", ylabel="coverage",
                title="P10-P90 coverage (target 0.80)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_attribution(attr, path, cp=1):
    g = attr[attr.cp == cp]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for k in range(4):
        axes[k].scatter(g[f"true_phi_stage{k}"], g[f"S{k}"], s=8, alpha=.5, color="#2563eb")
        axes[k].set_title(STAGE_NAMES[k]); axes[k].set_xlabel("true causal contribution (h)")
    axes[0].set_ylabel("explained contribution (h)")
    fig.suptitle(f"Explanation vs generator ground truth @ {CP_NAMES[cp]}", y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_missingness(sweep, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
    for ax, cp in zip(axes, (1, 2)):
        for mode, c in [("model", "#10b981"), ("locf", "#f59e0b")]:
            g = sweep[(sweep.cp == cp) & (sweep.imputation == mode)]
            ax.plot(g["extra_missing"], g["MAE"], "o-", color=c, label={"model": "stage-aware imputer", "locf": "carry-forward"}[mode])
        ax.set(title=CP_NAMES[cp], xlabel="extra missing-event rate", ylabel="MAE (h)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_breakdown(bd, path, title):
    """Stage breakdown of the chain model's expected final delay (observed vs expected stages)."""
    colors = {"observed": "#0f172a", "estimated": "#64748b", "expected": "#0f766e"}
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, r in enumerate(bd.itertuples()):
        ax.barh(i, r.delta_h, color=colors[r.status], label=r.status)
        ax.text(r.delta_h + (0.4 if r.delta_h >= 0 else -0.4), i, f"{r.delta_h:+.1f} h", va="center",
                ha="left" if r.delta_h >= 0 else "right", fontsize=9)
    ax.set_yticks(range(len(bd)), bd.stage); ax.invert_yaxis(); ax.axvline(0, color="k", lw=.6)
    h, l = ax.get_legend_handles_labels()
    ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), frameon=False, fontsize=8, loc="lower right")
    ax.set_xlabel("hours added during the stage (negative = buffer recovered)")
    ax.set_title(f"{title}\nexpected final delay = {bd.delta_h.sum():+.1f} h", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_pred_vs_actual(oof, path, model="delay_chain_median"):
    cps = sorted(oof.cp.unique())
    fig, axes = plt.subplots(1, len(cps), figsize=(4.2 * len(cps), 4.1), sharex=True, sharey=True)
    for ax, cp in zip(np.atleast_1d(axes), cps):
        g = oof[oof.cp == cp].groupby("shipment_id").agg(actual=("true_L3", "first"), pred=(model, "mean"),
                                                          sched=("delay_sched", "mean"))
        late = g.actual > DELAY_THRESHOLD_H
        ax.scatter(g.actual[~late], g.pred[~late], s=12, alpha=.6, color="#60a5fa", label="on time")
        ax.scatter(g.actual[late], g.pred[late], s=12, alpha=.7, color="#ef4444", label=f"delayed >{DELAY_THRESHOLD_H:.0f}h")
        lim = [min(g.actual.min(), g.pred.min()) - 2, max(g.actual.max(), g.pred.max()) + 2]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_title(f"{CP_NAMES[cp]}\nMAE {np.abs(g.pred - g.actual).mean():.1f}h "
                     f"(schedule alone: {np.abs(g.sched - g.actual).mean():.1f}h)", fontsize=9)
        ax.set_xlabel("actual final delay (h)")
    np.atleast_1d(axes)[0].set_ylabel("predicted final delay (h)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Predicted vs actual (out-of-sample). Closer to the dashed line = better", y=1.02, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_eta_convergence(oof, df, path, n=8, model="delay_chain_median"):
    """For a spread of shipments: how the predicted ETA moves toward the actual arrival as stages complete."""
    g = oof.groupby(["shipment_id", "cp"]).agg(pred=(model, "mean"), lo=("c10", "mean"), hi=("c90", "mean"),
                                               actual=("true_L3", "first")).reset_index()
    act = g.groupby("shipment_id").actual.first().sort_values()
    ids = list(act.index[np.linspace(0, len(act) - 1, min(n, len(act))).astype(int)])
    meta = df.set_index("shipment_id")
    colors = {0: "#9ca3af", 1: "#3b82f6", 2: "#10b981"}
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(ids) + 1.8))
    for i, sid in enumerate(ids):
        for cp in (0, 1, 2):
            r = g[(g.shipment_id == sid) & (g.cp == cp)].iloc[0]
            y = i + (cp - 1) * 0.24
            ax.plot([r.lo, r.hi], [y, y], color=colors[cp], lw=3.5, alpha=.4, solid_capstyle="round")
            ax.plot(r.pred, y, "o", color=colors[cp], ms=5)
        ax.plot(act[sid], i, "*", color="black", ms=13, zorder=5)
    ax.axvline(0, color="#374151", ls="--", lw=1)
    ax.set_yticks(range(len(ids)), [f"{s}  {meta.loc[s, 'origin']}→{meta.loc[s, 'destination']}" for s in ids], fontsize=8)
    ax.set_xlabel("final delay vs schedule (hours)   [0 = scheduled arrival]")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color=colors[c], label=f"predicted @ {CP_NAMES[c]}") for c in (0, 1, 2)]
              + [Line2D([], [], marker="*", color="black", ls="", ms=11, label="actual arrival"),
                 Line2D([], [], color="#374151", ls="--", label="scheduled")], fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Predicted ETA converges to the actual arrival as the shipment progresses (bar = 80% interval)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_route_mae(oof, df, path, model="delay_chain_median"):
    lab = df.drop_duplicates("route_id").set_index("route_id").pipe(lambda d: d.origin + "→" + d.destination)
    rows = []
    for (rid, cp), g in oof.groupby(["route_id", "cp"]):
        rows.append(dict(route=lab[rid], cp=cp, mae=(g[model] - g.true_L3).abs().mean(), sched=(g.true_L3).abs().mean()))
    t = pd.DataFrame(rows)
    piv = t.pivot(index="route", columns="cp", values="mae")
    piv["sched"] = t[t.cp == t.cp.min()].set_index("route")["sched"]
    piv = piv.sort_values("sched")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(piv) + 1.5))
    h, y = 0.2, np.arange(len(piv))
    ax.barh(y - h, piv["sched"], h, color="#9ca3af", label="schedule only")
    for k, cp in enumerate(sorted(c for c in piv.columns if c != "sched")):
        ax.barh(y + k * h, piv[cp], h, color=["#93c5fd", "#3b82f6", "#10b981"][k % 3], label=CP_NAMES[cp])
    ax.set_yticks(y, piv.index, fontsize=8); ax.set_xlabel("MAE of final ETA (hours)")
    ax.legend(frameon=False, fontsize=8); ax.set_title("ETA error by lane", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
