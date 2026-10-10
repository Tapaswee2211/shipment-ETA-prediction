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
