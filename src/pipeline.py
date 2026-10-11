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
    # Parse the arguments 
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_SHIPMENTS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--fast", action="store_true", help="fewer repeats (quick smoke run)")
    a = ap.parse_args()
    t0 = time.time()

    OUT_DIR.mkdir(exist_ok=True, parents=True)



if __name__ == "__main__":
    main()
