from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

from .config import APPT_WINDOW_H
from .generator import build_routes

H = pd.Timedelta(hours=1)
LANES = {(r["origin"], r["destination"]): r for r in build_routes()}
ORIGINS = sorted({o for o, _ in LANES})
CONGESTION_RANGE = (0.30, 0.97)       # range of the utilisation index seen in training
DWELL_TOLERANCE_H = 1.0


def destinations_for(origin: str) -> list[str]:
    return sorted(d for o, d in LANES if o == origin)


def lane_properties(origin: str, destination: str) -> dict:
    r = LANES[(origin, destination)]
    return dict(route_id=r["route_id"], sched_transit_h=r["T1"], slack_transit_h=r["s1"], sched_port_h=r["P"],
                slack_port_h=r["s2"], sched_inland_h=r["T3"], slack_inland_h=r["s3"], proc_nominal_h=r["proc"],
                berth_window_h=r["berth"], weather_exposure=r["exposure"])


def typical_conditions(df: pd.DataFrame, origin: str, destination: str) -> dict:
    """Data-derived defaults for the condition inputs (medians in the training data)."""
    return dict(dest_congestion=float(df.loc[df.destination == destination, "dest_congestion_arr"].median()),
                origin_congestion=float(df.loc[df.origin == origin, "origin_congestion"].median()),
                weather=float(df["weather_fcst"].median()))


# ----------------------------------------------------------------------------- existing shipments
def normalize_id(text) -> str:
    return str(text).strip().upper()


def holdout_split(df: pd.DataFrame, shipment_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training frame WITHOUT the shipment, the one-row shipment). Raises KeyError for an unknown id."""
    hit = df["shipment_id"].str.upper() == normalize_id(shipment_id)
    if not hit.any():
        raise KeyError(f"unknown shipment id '{shipment_id}'")
    return df[~hit], df[hit]


def as_of(row: pd.DataFrame, cp: int) -> pd.DataFrame:
    """Blank everything that is not knowable at checkpoint cp (later milestones, realised conditions, outcome,
    simulator ground truth)."""
    future = {"delivery_actual_ts"}
    if cp < 1:
        future |= {"port_arrival_actual_ts", "weather_actual", "dest_congestion_arr"}
    if cp < 2:
        future |= {"customs_cleared_actual_ts", "port_wait_h", "processing_h", "inspection_flag"}
    future |= {c for c in row.columns if c.startswith("true_")}
    r = row.copy()
    for c in future & set(r.columns):
        r[c] = pd.NaT if pd.api.types.is_datetime64_any_dtype(r[c]) else np.nan
    return r


# ----------------------------------------------------------------------------- new shipments
@dataclass
class NewShipment:
    ref: str
    origin: str
    destination: str
    dep_sched: pd.Timestamp                  # UTC
    cp: int                                  # 0 departed, 1 arrived at port, 2 customs cleared
    L0: float                                # hours late at departure (negative = early)
    origin_congestion: float
    weather_fcst: float
    dest_congestion_dep: float               # utilisation forecast known at departure
    L1: float | None = None                  # cp>=1: hours late at port arrival
    weather_actual: float | None = None
    dest_congestion_arr: float | None = None
    L2: float | None = None                  # cp>=2: hours late at customs clearance
    port_wait_h: float | None = None
    processing_h: float | None = None
    inspection: bool | None = None


class Validation(NamedTuple):
    errors: list
    warnings: list


def _ok(x) -> bool:
    return x is not None and np.isfinite(x)


def validate_new_shipment(inp: NewShipment, known_ids, train_dep_range=None) -> Validation:
    errs, warns = [], []
    ref = (inp.ref or "").strip()
    if not ref:
        errs.append("Shipment reference is required.")
    elif not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", ref):
        errs.append("Reference may only contain letters, digits, '-' and '_' (max 40 characters).")
    elif ref.upper() in {normalize_id(k) for k in known_ids}:
        errs.append(f"'{ref}' already exists in the dataset - use 'Existing shipment' mode for it.")
    if inp.cp not in (0, 1, 2):
        errs.append("Checkpoint must be 0 (departed), 1 (arrived at port) or 2 (customs cleared).")
    if (inp.origin, inp.destination) not in LANES:
        errs.append(f"Lane {inp.origin} -> {inp.destination} is not in the route table ({len(LANES)} supported lanes). "
                    "Its transit time, buffers and port parameters are unknown, so a prediction would be fabricated.")
    if not isinstance(inp.dep_sched, pd.Timestamp) or pd.isna(inp.dep_sched):
        errs.append("Scheduled departure must be a valid date and time (UTC).")
    if errs:
        return Validation(errs, warns)

    def rng(name, v, lo, hi):
        if not _ok(v):
            errs.append(f"{name} is required at this checkpoint.")
        elif not lo <= v <= hi:
            errs.append(f"{name} must be between {lo:g} and {hi:g} (got {v:g}).")

    c_lo, c_hi = CONGESTION_RANGE
    rng("Delay at departure (h)", inp.L0, -6, 72)
    rng("Origin congestion", inp.origin_congestion, 0, 1)
    rng("Weather forecast", inp.weather_fcst, 0, 1)
    rng("Destination congestion forecast", inp.dest_congestion_dep, c_lo, c_hi)
    if inp.cp >= 1:
        rng("Delay at port arrival (h)", inp.L1, -24, 150)
        rng("Weather encountered", inp.weather_actual, 0, 1)
        rng("Destination congestion at arrival", inp.dest_congestion_arr, c_lo, c_hi)
    if inp.cp >= 2:
        rng("Delay at customs clearance (h)", inp.L2, -24, 200)
        rng("Queue wait (h)", inp.port_wait_h, 0, 200)
        rng("Processing time (h)", inp.processing_h, 0, 200)
        if inp.inspection is None:
            errs.append("Inspection flag is required at this checkpoint.")
    if errs:
        return Validation(errs, warns)

    # physical consistency of the observed milestones
    lane = lane_properties(inp.origin, inp.destination)
    dep_act = inp.dep_sched + inp.L0 * H
    port_sched = inp.dep_sched + lane["sched_transit_h"] * H
    if inp.cp >= 1:
        port_act = port_sched + inp.L1 * H
        if port_act <= dep_act:
            errs.append("Port arrival would be before departure - check the delay at departure and at port arrival.")
        if inp.cp >= 2:
            cus_act = port_sched + (lane["sched_port_h"] + inp.L2) * H
            if cus_act <= port_act:
                errs.append("Customs clearance would be before port arrival - check the two delays.")
            else:
                dwell = (cus_act - port_act) / H
                need = inp.port_wait_h + inp.processing_h
                if dwell + DWELL_TOLERANCE_H < need:
                    errs.append(f"Queue wait + processing ({need:.1f} h) is longer than the port dwell implied by your "
                                f"arrival and clearance delays ({dwell:.1f} h). Raise the clearance delay or shorten them.")
    if train_dep_range is not None and not (train_dep_range[0] <= inp.dep_sched <= train_dep_range[1]):
        warns.append(f"Departure is outside the training period ({train_dep_range[0]:%d %b %Y} - "
                     f"{train_dep_range[1]:%d %b %Y}); calendar effects (weekends, holidays) may be less reliable.")
    if inp.dep_sched.year != 2025:
        warns.append("The holiday calendar used by the model only covers 2025.")
    return Validation(errs, warns)


def build_row(inp: NewShipment, template: pd.DataFrame) -> pd.DataFrame:
    """One row in the training schema. Only fields observable at inp.cp are filled; the rest stay NaN/NaT."""
    lane = lane_properties(inp.origin, inp.destination)
    row = {c: (pd.NaT if pd.api.types.is_datetime64_any_dtype(template[c]) else np.nan) for c in template.columns}
    row.update(lane)
    port_sched = inp.dep_sched + lane["sched_transit_h"] * H
    cus_sched = port_sched + lane["sched_port_h"] * H
    row.update(shipment_id=inp.ref.strip(), origin=inp.origin, destination=inp.destination,
               appt_window_h=APPT_WINDOW_H, dep_sched_ts=inp.dep_sched, port_arr_sched_ts=port_sched,
               customs_sched_ts=cus_sched, delivery_sched_ts=cus_sched + lane["sched_inland_h"] * H,
               dep_actual_ts=inp.dep_sched + inp.L0 * H, origin_congestion=inp.origin_congestion,
               weather_fcst=inp.weather_fcst, dest_congestion_dep=inp.dest_congestion_dep)
    if inp.cp >= 1:
        row.update(port_arrival_actual_ts=port_sched + inp.L1 * H, weather_actual=inp.weather_actual,
                   dest_congestion_arr=inp.dest_congestion_arr)
    if inp.cp >= 2:
        row.update(customs_cleared_actual_ts=cus_sched + inp.L2 * H, port_wait_h=inp.port_wait_h,
                   processing_h=inp.processing_h, inspection_flag=float(bool(inp.inspection)))
    out = pd.DataFrame([row], columns=list(template.columns))
    for c in template.columns:
        if pd.api.types.is_datetime64_any_dtype(template[c]):
            out[c] = pd.to_datetime(out[c])
    return out
