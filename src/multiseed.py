import argparse

import pandas as pd

from .config import CP_NAMES, N_SHIPMENTS, OUT_DIR
from .evaluate import reg_metrics, run_cv
from .generator import generate_dataset

MODELS = ["delay_sched", "delay_carry", "delay_gbm", "delay_chain", "delay_chain_median"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=N_SHIPMENTS)
    ap.add_argument("--first", type=int, default=1, help="first seed")
    ap.add_argument("--split", default="random", choices=["random", "route", "time"])
    a = ap.parse_args()
    rows = []
    for seed in range(a.first, a.first + a.seeds):
        df = generate_dataset(a.n, seed)
        m = reg_metrics(run_cv(df, a.split, full=True, seed=seed, n_repeats=1), MODELS)
        rows += [dict(seed=seed, cp=int(r.cp), model=r.model, MAE=r.MAE) for r in m.itertuples()]
        print(f"seed {seed} done", flush=True)
    t = pd.DataFrame(rows)
    g = t.groupby(["cp", "model"]).MAE.agg(["mean", "std"]).reset_index()
    g["MAE (mean ± std over seeds)"] = [f"{r['mean']:.2f} ± {r['std']:.2f}" for _, r in g.iterrows()]
    out = g.pivot(index="model", columns="cp", values="MAE (mean ± std over seeds)").rename(columns=CP_NAMES)
    out = out.loc[[m.replace("delay_", "") for m in MODELS]]
    lines = [f"MAE in hours, seeds {a.first}..{a.first + a.seeds - 1}, {a.split} split, n={a.n}", "", "| model | " + " | ".join(out.columns) + " |",
             "|---|" + "---|" * len(out.columns)]
    lines += [f"| {i} | " + " | ".join(r) + " |" for i, r in out.iterrows()]
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    (OUT_DIR / ("multiseed.md" if (a.split, a.first) == ("random", 1) else f"multiseed_{a.split}_from{a.first}.md")).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    wins = (t[t.model == "chain_median"].set_index(["seed", "cp"]).MAE < t[t.model == "carry"].set_index(["seed", "cp"]).MAE)
    print(f"\nchain_median beats carry-forward in {int(wins.sum())}/{len(wins)} (seed, checkpoint) cells")


if __name__ == "__main__":
    main()
