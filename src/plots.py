import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CP_NAMES, DELAY_THRESHOLD_H
from .explain import STAGE_NAMES

COL = {"sched": "#9ca3af", "carry": "#f59e0b", "route_prior": "#a78bfa", "ridge": "#60a5fa",
       "gbm": "#10b981", "chain": "#ef4444"}


def fig_mae_by_checkpoint(reg, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    cps = sorted(reg["cp"].unique())
    w = 0.13
    for i, m in enumerate(["sched", "carry", "route_prior", "ridge", "gbm", "chain"]):
        g = reg[reg.model == m].sort_values("cp")
        if g.empty:
            continue
        x = np.arange(len(cps)) + (i - 2.5) * w
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


def fig_waterfall(A_row, pred_total, path, title):
    labels = [STAGE_NAMES[k] for k in range(4)] + ["baseline +\npropagation"]
    vals = [A_row[f"S{k}"] for k in range(4)] + [A_row["other"]]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    cum = 0
    for i, v in enumerate(vals):
        ax.bar(i, v, bottom=cum, color="#ef4444" if v > 0 else "#10b981")
        ax.text(i, cum + v + (0.5 if v >= 0 else -1.5), f"{v:+.1f}", ha="center", fontsize=9)
        cum += v
    ax.bar(len(vals), cum, color="#1f2937")
    ax.text(len(vals), cum + 0.5, f"{cum:+.1f} h", ha="center", fontsize=9)
    ax.set_xticks(range(len(vals) + 1), labels + ["predicted\nfinal delay"], fontsize=8)
    ax.axhline(0, color="k", lw=.5); ax.set_ylabel("hours"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
