from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .config import APPT_WINDOW_H
from .features import set_holidays

H = pd.Timedelta(hours=1)

TS_COLS = ["dep_sched_ts", "port_arr_sched_ts", "customs_sched_ts", "delivery_sched_ts",
           "dep_actual_ts", "port_arrival_actual_ts", "customs_cleared_actual_ts", "delivery_actual_ts"]

# canonical name -> (required?, meaning)
SCHEMA = {
    "origin": (True, "origin port / city"),
    "destination": (True, "destination port / city"),
    "dep_sched_ts": (True, "scheduled departure (ETD)"),
    "dep_actual_ts": (True, "actual departure (ATD)"),
    "delivery_sched_ts": (True, "scheduled final delivery"),
    "delivery_actual_ts": (True, "actual final delivery (the prediction target)"),
    "port_arr_sched_ts": (True, "scheduled port arrival (derivable via sched_transit_h or stage_split)"),
    "port_arrival_actual_ts": (True, "actual port arrival; may be empty for some rows"),
    "customs_sched_ts": (True, "scheduled customs/port clearance (derivable via sched_port_h or stage_split)"),
    "customs_cleared_actual_ts": (True, "actual customs/port clearance; may be empty for some rows"),
    "shipment_id": (False, "unique id (generated if absent)"),
    "route_id": (False, "lane id (origin->destination if absent)"),
    "sched_transit_h": (False, "scheduled ocean transit hours (derived from timestamps)"),
    "sched_port_h": (False, "scheduled port dwell hours (derived)"),
    "sched_inland_h": (False, "scheduled inland hours (derived)"),
    "slack_transit_h": (False, "buffer built into the transit leg (hours)"),
    "slack_port_h": (False, "buffer built into the port leg"),
    "slack_inland_h": (False, "buffer built into the inland leg"),
    "proc_nominal_h": (False, "typical port processing/dwell hours (derived from history)"),
    "berth_window_h": (False, "lateness beyond which the berth slot is lost"),
    "appt_window_h": (False, "lateness beyond which the delivery appointment is lost"),
    "weather_exposure": (False, "0-1 lane exposure to bad weather"),
    "weather_fcst": (False, "0-1 weather severity forecast at departure"),
    "weather_actual": (False, "0-1 weather severity realised in transit"),
    "origin_congestion": (False, "0-1 origin port utilisation at departure"),
    "dest_congestion_dep": (False, "0-1 destination utilisation known at departure"),
    "dest_congestion_arr": (False, "0-1 destination utilisation at arrival"),
    "port_wait_h": (False, "observed queue wait before processing"),
    "processing_h": (False, "observed port processing hours (derived from timestamps)"),
    "inspection_flag": (False, "1 if the box was inspected"),
}


def describe_schema() -> str:
    return "\n".join(f"{'REQ' if r else 'opt'}  {k:<26} {d}" for k, (r, d) in SCHEMA.items())


def _to_utc_naive(s: pd.Series, fmt, tz) -> pd.Series:
    if tz in (None, "UTC", "utc"):
        t = pd.to_datetime(s, format=fmt, errors="coerce", utc=True)
    else:
        t = pd.to_datetime(s, format=fmt, errors="coerce")
        if getattr(t.dt, "tz", None) is None:
            t = t.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
        t = t.dt.tz_convert("UTC")
    return t.dt.tz_localize(None)


def standardize(raw: pd.DataFrame, mapping: dict | None = None, min_rows: int = 60, min_events: int = 30):
    """Map a raw table onto the canonical schema. Returns (clean_df, notes). Raises ValueError with a clear message."""
    m = mapping or {}
    notes: list[str] = []
    df = pd.DataFrame(index=raw.index)
    for canon, src in m.get("columns", {}).items():
        if canon not in SCHEMA:
            raise ValueError(f"unknown canonical column '{canon}'. Valid: {', '.join(SCHEMA)}")
        if src not in raw.columns:
            raise ValueError(f"mapped column '{src}' (for '{canon}') not found. Your columns: {list(raw.columns)}")
        df[canon] = raw[src]
    for c in raw.columns:                                   # already-canonical columns pass through
        if c in SCHEMA and c not in df:
            df[c] = raw[c]
    for k, v in m.get("constants", {}).items():
        df[k] = v

    for c in TS_COLS:
        if c in df:
            df[c] = _to_utc_naive(df[c], m.get("timestamp_format"), m.get("timezone"))

    must = ["origin", "destination", "dep_sched_ts", "dep_actual_ts", "delivery_sched_ts", "delivery_actual_ts",
            "port_arrival_actual_ts", "customs_cleared_actual_ts"]
    miss = [c for c in must if c not in df]
    if miss:
        raise ValueError(f"missing required columns {miss}. Map them in 'columns'. Schema:\n{describe_schema()}")

    # --- scheduled intermediate milestones: use them, else derive
    tot = df["delivery_sched_ts"] - df["dep_sched_ts"]
    split = m.get("stage_split")
    if split:
        split = np.asarray(split, float) / np.sum(split)
    if "port_arr_sched_ts" not in df or df["port_arr_sched_ts"].isna().all():
        if "sched_transit_h" in df and df["sched_transit_h"].notna().any():
            df["port_arr_sched_ts"] = df["dep_sched_ts"] + pd.to_timedelta(df["sched_transit_h"].astype(float), unit="h")
        elif split is not None:
            df["port_arr_sched_ts"] = df["dep_sched_ts"] + tot * split[0]
            notes.append(f"port_arr_sched_ts derived from stage_split={split.round(2).tolist()}")
        else:
            raise ValueError("no scheduled port-arrival time: map 'port_arr_sched_ts', give 'sched_transit_h', "
                             "or set 'stage_split' in the mapping")
    if "customs_sched_ts" not in df or df["customs_sched_ts"].isna().all():
        if "sched_port_h" in df and df["sched_port_h"].notna().any():
            df["customs_sched_ts"] = df["port_arr_sched_ts"] + pd.to_timedelta(df["sched_port_h"].astype(float), unit="h")
        elif split is not None:
            df["customs_sched_ts"] = df["port_arr_sched_ts"] + tot * split[1]
            notes.append("customs_sched_ts derived from stage_split")
        else:
            raise ValueError("no scheduled customs time: map 'customs_sched_ts', give 'sched_port_h', or set 'stage_split'")

    # --- row hygiene
    n0 = len(df)
    req_ts = ["dep_sched_ts", "dep_actual_ts", "delivery_sched_ts", "delivery_actual_ts", "port_arr_sched_ts", "customs_sched_ts"]
    ok = df[req_ts].notna().all(axis=1)
    ok &= (df["dep_sched_ts"] < df["port_arr_sched_ts"]) & (df["port_arr_sched_ts"] < df["customs_sched_ts"]) \
        & (df["customs_sched_ts"] < df["delivery_sched_ts"])
    if (~ok).any():
        notes.append(f"dropped {int((~ok).sum())}/{n0} rows with missing required timestamps or non-increasing schedule")
    df = df[ok].copy()

    # --- derived columns
    if "shipment_id" not in df:
        df["shipment_id"] = [f"S{i:05d}" for i in range(len(df))]
    if df["shipment_id"].duplicated().any():
        raise ValueError("shipment_id must be unique")
    if "route_id" not in df:
        df["route_id"] = df["origin"].astype(str) + "->" + df["destination"].astype(str)
    df["sched_transit_h"] = (df["port_arr_sched_ts"] - df["dep_sched_ts"]) / H
    df["sched_port_h"] = (df["customs_sched_ts"] - df["port_arr_sched_ts"]) / H
    df["sched_inland_h"] = (df["delivery_sched_ts"] - df["customs_sched_ts"]) / H
    dwell = (df["customs_cleared_actual_ts"] - df["port_arrival_actual_ts"]) / H
    if "proc_nominal_h" not in df:
        med = dwell.groupby(df["route_id"]).transform("median")
        df["proc_nominal_h"] = med.fillna(dwell.median()).fillna(df["sched_port_h"])
        notes.append("proc_nominal_h derived as the route-median observed port dwell")
    if "processing_h" not in df:
        df["processing_h"] = dwell
    df["appt_window_h"] = df["appt_window_h"] if "appt_window_h" in df else APPT_WINDOW_H
    absent = []
    for c, (req, _) in SCHEMA.items():
        if c not in df:
            df[c] = np.nan                                   # optional signal not available -> unknown
            absent.append(c)
    if absent:
        notes.append("optional columns not provided (treated as unknown): " + ", ".join(absent))

    # --- sanity
    n_arr, n_cus = int(df["port_arrival_actual_ts"].notna().sum()), int(df["customs_cleared_actual_ts"].notna().sum())
    if len(df) < min_rows:
        raise ValueError(f"only {len(df)} usable rows (need >= {min_rows})")
    if n_arr < min_events or n_cus < min_events:
        raise ValueError(f"need >= {min_events} rows with observed port-arrival and customs timestamps "
                         f"(have {n_arr} / {n_cus}). Map the closest real events to these milestones.")
    L3 = (df["delivery_actual_ts"] - df["delivery_sched_ts"]) / H
    if L3.abs().median() > 24 * 30:
        notes.append("WARNING: median |final lateness| > 30 days - check timestamp units / timezone / mapping")
    notes.append(f"{len(df)} shipments, {df['route_id'].nunique()} lanes; observed port-arrival {n_arr/len(df):.0%}, "
                 f"customs {n_cus/len(df):.0%}; final lateness mean {L3.mean():.1f}h, median {L3.median():.1f}h")
    if m.get("holidays"):
        set_holidays(m["holidays"])
        notes.append(f"holiday calendar set ({len(m['holidays'])} dates)")
    elif df["dep_sched_ts"].dt.year.min() != 2025 or df["dep_sched_ts"].dt.year.max() != 2025:
        notes.append("NOTE: default holiday calendar covers 2025 only; add 'holidays' to the mapping for other years")
    df = df.sort_values("dep_sched_ts").reset_index(drop=True)
    return df, notes


def load_dataset(path: str, mapping_path: str | None = None, **kw):
    raw = pd.read_csv(path)
    mapping = json.load(open(mapping_path, encoding="utf-8")) if mapping_path else None
    return standardize(raw, mapping, **kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="validate a dataset against the canonical schema")
    ap.add_argument("--data"); ap.add_argument("--mapping"); ap.add_argument("--schema", action="store_true")
    a = ap.parse_args()
    if a.schema or not a.data:
        print(describe_schema())
    if a.data:
        d, notes = load_dataset(a.data, a.mapping)
        print("\n".join(notes))
