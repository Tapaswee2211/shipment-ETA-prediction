from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from .generator import BASE, BASE_DOW, HOLIDAY_DAYS

H = pd.Timedelta(hours=1)
_HOL = np.array(sorted(HOLIDAY_DAYS))


def set_holidays(dates) -> None:
    """Replace the holiday calendar (list of date strings). Needed for real data from other years/countries."""
    global _HOL
    _HOL = np.array(sorted((pd.Timestamp(d) - BASE).days for d in dates), dtype=int)

STATIC_COLS = ["sched_transit_h", "slack_transit_h", "sched_port_h", "slack_port_h", "sched_inland_h",
               "slack_inland_h", "proc_nominal_h", "berth_window_h", "weather_exposure"]

# ---- feature groups (used for SHAP aggregation and ablations) ---------------
FEATURE_GROUP: dict[str, str] = {}


def _g(group, cols):
    for c in cols:
        FEATURE_GROUP[c] = group


_g("origin", ["L0", "origin_congestion"])
_g("weather", ["weather_fcst", "weather_actual", "weather_x_exposure_fcst"])
_g("transit_obs", ["inc_transit"])
_g("congestion", ["dest_congestion_dep", "queue_dep", "x_cong_proc_dep", "dest_congestion_arr", "queue_arr",
                  "x_cong_proc_arr", "x_cong_weekend"])
_g("port_obs", ["inc_port", "port_wait_h", "processing_h", "proc_excess", "inspection_flag"])
_g("calendar", ["dep_dow", "dep_hour", "sched_arr_dow", "sched_arr_weekend", "sched_arr_holiday", "arr_dow",
                "arr_weekend", "arr_holiday"])
_g("carried", ["L1", "L2", "buffer_left_transit", "buffer_left_port", "buffer_left_inland", "berth_risk", "appt_risk"])
_g("route_static", STATIC_COLS + ["prior_route_delay", "prior_route_std"])
_g("data_quality", ["L1_missing", "L2_missing"])

INTERACTION_COLS = ["x_cong_proc_dep", "x_cong_proc_arr", "x_cong_weekend", "weather_x_exposure_fcst",
                    "queue_dep", "queue_arr"]
PROPAGATION_COLS = ["inc_transit", "inc_port", "buffer_left_transit", "buffer_left_port", "buffer_left_inland",
                    "berth_risk", "appt_risk"]


# ---- time helpers -----------------------------------------------------------
def hours_since_base(ts: pd.Series) -> pd.Series:
    return (ts - BASE) / H


def dow_from_hours(h):
    return (BASE_DOW + np.floor(np.asarray(h, dtype=float) / 24.0)) % 7


def holiday_from_hours(h):
    return np.isin(np.floor(np.asarray(h, dtype=float) / 24.0).astype(int), _HOL).astype(float)


def queue_factor(c):
    c = np.clip(np.asarray(c, dtype=float), 0.0, 0.97)
    return c / (1.0 - c)          # M/M/1-style waiting-time multiplier: explodes as utilisation -> 1


def add_lateness(df: pd.DataFrame) -> pd.DataFrame:
    """Observed lateness per milestone from timestamps. NaN where the timestamp was not recorded."""
    d = df.copy()
    d["L0"] = (d["dep_actual_ts"] - d["dep_sched_ts"]) / H
    d["L1_obs"] = (d["port_arrival_actual_ts"] - d["port_arr_sched_ts"]) / H
    d["L2_obs"] = (d["customs_cleared_actual_ts"] - d["customs_sched_ts"]) / H
    if "delivery_actual_ts" in d:
        d["L3"] = (d["delivery_actual_ts"] - d["delivery_sched_ts"]) / H
    return d


# ---- missing-data handling -------------------------------------------------
class LatenessImputer:
    """
    Fills missing L1 / L2 (un-logged milestones) with a small model of the *stage increment*
    given what is known at the start of that stage -- i.e. it exploits stage structure.
    mode='locf' is the naive baseline: carry the last observed lateness forward.
    """

    def __init__(self, mode: str = "model", seed: int = 0):
        self.mode, self.seed = mode, seed

    @staticmethod
    def _X1(d, L0):
        w = d["weather_actual"].to_numpy(float)
        return np.c_[L0, w, w * d["weather_exposure"].to_numpy(float), d["slack_transit_h"].to_numpy(float),
                     d["sched_transit_h"].to_numpy(float)]

    @staticmethod
    def _X2(d, L1):
        c = np.clip(d["dest_congestion_arr"].to_numpy(float), 0, 0.97)
        q = queue_factor(c)
        h = hours_since_base(d["port_arr_sched_ts"]).to_numpy(float) + L1
        return np.c_[L1, c, q, q * d["proc_nominal_h"].to_numpy(float), dow_from_hours(h) >= 5,
                     holiday_from_hours(h), L1 - d["berth_window_h"].to_numpy(float),
                     d["slack_port_h"].to_numpy(float)]

    def _hgb(self):
        return HistGradientBoostingRegressor(max_depth=3, max_iter=100, learning_rate=0.08,
                                             min_samples_leaf=8, random_state=self.seed)

    def fit(self, d: pd.DataFrame):
        if self.mode != "model":
            return self
        L0 = d["L0"].to_numpy(float)
        o1 = d["L1_obs"].to_numpy(float)
        m1 = ~np.isnan(o1)
        self.m1 = self._hgb().fit(self._X1(d, L0)[m1], (o1 - L0)[m1])
        L1f = np.where(m1, o1, L0 + self.m1.predict(self._X1(d, L0)))
        o2 = d["L2_obs"].to_numpy(float)
        m2 = ~np.isnan(o2)
        self.m2 = self._hgb().fit(self._X2(d, L1f)[m2], (o2 - L1f)[m2])
        return self

    def transform(self, d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        L0 = d["L0"].to_numpy(float)
        o1, o2 = d["L1_obs"].to_numpy(float), d["L2_obs"].to_numpy(float)
        d["L1_missing"] = np.isnan(o1).astype(float)
        d["L2_missing"] = np.isnan(o2).astype(float)
        if self.mode == "model":
            L1 = np.where(np.isnan(o1), L0 + self.m1.predict(self._X1(d, L0)), o1)
            L2 = np.where(np.isnan(o2), L1 + self.m2.predict(self._X2(d, L1)), o2)
        else:                                            # last-observation-carried-forward
            L1 = np.where(np.isnan(o1), L0, o1)
            L2 = np.where(np.isnan(o2), L1, o2)
        d["L1"], d["L2"] = L1, L2
        return d


# ---- route priors (target encoding done right) ----------------------------
class RoutePriors:
    """Shrunk per-route mean/std of final delay. Training rows get *out-of-fold* priors to avoid leakage."""

    def __init__(self, k: float = 8.0, folds: int = 5, seed: int = 0):
        self.k, self.folds, self.seed = k, folds, seed

    def _stats(self, d, gm, gs):
        out = {}
        for rid, g in d.groupby("route_id")["L3"]:
            n = len(g)
            out[rid] = ((g.sum() + self.k * gm) / (n + self.k),
                        (g.std(ddof=0) * n + self.k * gs) / (n + self.k))
        return out

    @staticmethod
    def _apply(d, stats, gm, gs):
        arr = np.array([stats.get(r, (gm, gs)) for r in d["route_id"]], dtype=float)
        return arr.reshape(len(d), 2)

    def fit(self, d):
        self.gm, self.gs = float(d["L3"].mean()), float(d["L3"].std(ddof=0))
        self.stats = self._stats(d, self.gm, self.gs)
        return self

    def transform(self, d):
        return self._apply(d, self.stats, self.gm, self.gs)

    def fit_transform(self, d):
        self.fit(d)
        out = np.zeros((len(d), 2))
        for tr, te in KFold(self.folds, shuffle=True, random_state=self.seed).split(d):
            dtr = d.iloc[tr]
            gm, gs = float(dtr["L3"].mean()), float(dtr["L3"].std(ddof=0))
            out[te] = self._apply(d.iloc[te], self._stats(dtr, gm, gs), gm, gs)
        return out


# ---- main feature builder --------------------------------------------------
def build_features(d: pd.DataFrame, cp: int) -> pd.DataFrame:
    """d must contain imputed L1/L2, missing flags and prior_route_* columns."""
    X = pd.DataFrame(index=d.index)
    for c in STATIC_COLS:
        X[c] = d[c].astype(float)
    hd = hours_since_base(d["dep_sched_ts"])
    ha = hours_since_base(d["port_arr_sched_ts"])
    X["dep_dow"] = dow_from_hours(hd)
    X["dep_hour"] = d["dep_sched_ts"].dt.hour.astype(float)
    X["sched_arr_dow"] = dow_from_hours(ha)
    X["sched_arr_weekend"] = (X["sched_arr_dow"] >= 5).astype(float)
    X["sched_arr_holiday"] = holiday_from_hours(ha)
    X["origin_congestion"] = d["origin_congestion"]
    X["L0"] = d["L0"]
    X["buffer_left_transit"] = d["slack_transit_h"] - np.maximum(d["L0"], 0)
    X["dest_congestion_dep"] = d["dest_congestion_dep"]
    qd = queue_factor(d["dest_congestion_dep"])
    X["queue_dep"] = qd
    X["weather_fcst"] = d["weather_fcst"]
    X["weather_x_exposure_fcst"] = d["weather_fcst"] * d["weather_exposure"]
    X["x_cong_proc_dep"] = qd * d["proc_nominal_h"]
    X["prior_route_delay"] = d["prior_route_delay"]
    X["prior_route_std"] = d["prior_route_std"]
    if cp >= 1:
        X["L1"] = d["L1"]
        X["inc_transit"] = d["L1"] - d["L0"]
        X["L1_missing"] = d["L1_missing"]
        X["weather_actual"] = d["weather_actual"]
        X["dest_congestion_arr"] = d["dest_congestion_arr"]
        qa = queue_factor(d["dest_congestion_arr"])
        X["queue_arr"] = qa
        X["x_cong_proc_arr"] = qa * d["proc_nominal_h"]
        harr = ha + d["L1"]
        X["arr_dow"] = dow_from_hours(harr)
        X["arr_weekend"] = (X["arr_dow"] >= 5).astype(float)
        X["arr_holiday"] = holiday_from_hours(harr)
        X["x_cong_weekend"] = qa * X["arr_weekend"]
        X["berth_risk"] = d["L1"] - d["berth_window_h"]
        X["buffer_left_port"] = d["slack_port_h"] - np.maximum(d["L1"], 0)
    if cp >= 2:
        X["L2"] = d["L2"]
        X["inc_port"] = d["L2"] - d["L1"]
        X["L2_missing"] = d["L2_missing"]
        X["port_wait_h"] = d["port_wait_h"]
        X["processing_h"] = d["processing_h"]
        X["proc_excess"] = d["processing_h"] - d["proc_nominal_h"]
        X["inspection_flag"] = d["inspection_flag"]
        X["appt_risk"] = d["L2"] - d["appt_window_h"]
        X["buffer_left_inland"] = d["slack_inland_h"] - np.maximum(d["L2"], 0)
    return X.astype(float)
