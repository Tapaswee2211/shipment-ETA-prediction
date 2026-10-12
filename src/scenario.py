from __future__ import annotations

import numpy as np
import pandas as pd

WHATIF_FIELDS = {0: {"congestion": "dest_congestion_dep", "weather": "weather_fcst"},
                 1: {"congestion": "dest_congestion_arr"},
                 2: {}}


def available(cp: int) -> dict:
    return WHATIF_FIELDS[cp]


def apply_whatif(row: pd.DataFrame, cp: int, congestion: float | None = None, weather: float | None = None) -> pd.DataFrame:
    fields, r = WHATIF_FIELDS[cp], row.copy()
    for name, value in (("congestion", congestion), ("weather", weather)):
        if value is None:
            continue
        if name not in fields:
            raise ValueError(f"'{name}' cannot influence the remaining stages at checkpoint {cp}")
        r[fields[name]] = value
    return r


def sweep(P, row: pd.DataFrame, kind: str, grid) -> pd.DataFrame:
    """Median / 80% range / P(delay) of the chain model as one condition varies. Common random numbers make the
    curve free of sampling noise."""
    col = WHATIF_FIELDS[P.cp][kind]
    rows = pd.concat([row] * len(grid), ignore_index=True)
    rows[col] = np.asarray(grid, float)
    o = P.predict(rows)
    return pd.DataFrame({"x": np.asarray(grid, float), "median": o["delay_chain_median"].to_numpy(),
                         "lo": o["c10"].to_numpy(), "hi": o["c90"].to_numpy(), "p": o["p_delay_chain"].to_numpy()})
