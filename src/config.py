""" Global Configuration """
 
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

SEED = 54
N_SHIPMENTS = 300

"""
Lateness can be stored with L_k = actual arrival time at k - scheduled_k (hours, UTC). 

here k0 => Departure
k1 => Port Arrival
k2 => Customs Cleared
k3 => Final Delivery

"""
#Delayed can be a binary lable => L_3 > DELAY_THRESHOLD_H
DELAY_THRESHOLD_H = 12.0 
APPT_WINDOW_H = 20.0  # appointment for trucks : > than this => re-booking penalty 

ABSORB_EFF = 0.7 #Share of schedule slack that can actually be used to recover time 

CP_NAMES = {0: "CP0 departure", 1: "CP1 port arrival", 2: "CP2 customs cleared"}
CHECKPOINTS = (0, 1, 2)      # prediction is re-issued at each milestone
