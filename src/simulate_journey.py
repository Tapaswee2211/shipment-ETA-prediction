from __future__ import annotations
from math import factorial 
import numpy as np 
import pandas as pd 


from .config import ABSORB_EFF, APPT_WINDOW_H, DATA_DIR, N_SHIPMENTS, SEED

BASE = pd.Timestamp("2025-01-01")
BASE_DOW = BASE.dayofweek 

############## Set Holidays
_HOL = ["2025-01-01", "2025-04-18", "2025-04-21", "2025-05-01", "2025-05-29", "2025-12-25"]
HOLIDAYS = frozenset((pd.Timestamp(d) - BASE).days for d in _HOL)

# destination port: mean congestion level, nominal processing hours, berth-window hours
PORTS = {
    "Rotterdam":   dict(mu=0.0,  proc=24, berth=18),
    "Hamburg":     dict(mu=-0.2, proc=28, berth=18),
    "Los Angeles": dict(mu=0.6,  proc=36, berth=14),
    "Jebel Ali":   dict(mu=-0.5, proc=18, berth=20),
    "Felixstowe":  dict(mu=0.2,  proc=30, berth=16),
    "Singapore":   dict(mu=-0.4, proc=14, berth=20),
    "Savannah":    dict(mu=0.3,  proc=32, berth=14),
}

POOR_REPORTING = {"Jebel Ali", "Savannah"}
WAIT_SCALE = 1.5

# origin, dest, transit_h, slack_transit, slack_port, inland_h, slack_inland, weather_exposure, origin_congestion_mean
ROUTE_TABLE = [
    ("Shanghai",    "Rotterdam",   816, 36, 14, 48, 8, 0.9, 0.55),
    ("Nhava Sheva", "Hamburg",     576, 30, 12, 56, 8, 0.7, 0.45),
    ("Shenzhen",    "Los Angeles", 456, 30, 16, 40, 8, 1.0, 0.60),
    ("Busan",       "Los Angeles", 360, 24, 16, 40, 8, 1.0, 0.40),
    ("Nhava Sheva", "Jebel Ali",   120, 10, 10, 24, 6, 0.3, 0.45),
    ("Colombo",     "Felixstowe",  600, 30, 12, 48, 8, 0.8, 0.35),
    ("Chennai",     "Singapore",   168, 12,  8, 24, 6, 0.4, 0.40),
    ("Ningbo",      "Savannah",    912, 40, 14, 60, 10, 1.0, 0.50),
    ("Shanghai",    "Singapore",   144, 10,  8, 24, 6, 0.5, 0.55),
    ("Nhava Sheva", "Rotterdam",   624, 30, 14, 48, 8, 0.8, 0.45),
]


def build_routes() -> list[dict]:
    routes = []
    for i, (o, d, t1, s1, s2, t3, s3, weather_expo, ocm) in enumerate(ROUTE_TABLE, 1):
        p = PORTS[d]
        routes.append(dict(route_id=f"R{i:02d}", origin=o, destination=d, 
                        T1 = float(t1), s1=float(s1), s2=float(s2), T3=float(t3),s3=float(s3),
                        exposure=weather_expo, oc_mean=ocm, 
                        proc=float(p["proc"]),P=float(p["proc"] + s2), 
                        berth = float(p["berth"]), sigma=0.012 * t1 + 2.0)
                    )
    return routes

def _port_series(rng: np.random.Generator, n_days: int= 360) -> dict[str, np.ndarray]:
    out = {}
    for name, p in PORTS.items():
        x = np.zeros(n_days)
        x[0] = rng.normal(0, 0.9)
        for t in range(1, n_days):
            x[t] = 0.96 * x[t-1] + 0.25 * rng.normal()
        out[name] = 0.35 + 0.60 / (1.0 + np.exp(-(p["mu"] + x)))
    return out

def _dow ( h : float) -> int:
    return (BASE_DOW + int(h//24)) % 7


if __name__ == '__main__': 
    routes = build_routes()
    print(len(routes))      # Expected: 10
    print(routes[0])        # Shanghai -> Rotterdam
    print(routes[-1])      
