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

SOURCES = ["origin", "weather", "route_noise", "congestion", "processing", "inspection", "inland"]
SOURCE_STAGE = {"origin": 0, "weather": 1, "route_noise": 1, "congestion": 2,
                "processing": 2, "inspection": 2, "inland": 3}

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


def propagate (p: dict, r: dict, u: np.ndarray, zero: frozenset = frozenset()) -> dict :
    def on(g):
        return 0.0 if g in zero else 1.0 
    e = ABSORB_EFF

    #################### S0 : At origin
    L0 = on("origin") * p["o_flag"] * p["o_size"] * (0.5 + p["oc"])
    weather = on("weather") * 40.0 * r["exposure"] * p["w"]  ** 1.5
    rnoise = on("route_noise") * p["n1"] * r["sigma"]
    raw1 = L0 + weather + rnoise

    #################### S1 : Ocean transit
    L1 = raw1 - e * min(r["s1"], max(raw1, 0.0)) 

    #################### S2 : Queue at PORT and customs
    t_arr = p["dep_h"] + r["T1"] + L1
    uu = float(u[int(t_arr // 24)])
    v = (uu - 0.35) / 0.60
    cong = on("congestion")
    wait = cong * WAIT_SCALE * uu / (1.0 - uu) * p["qn"]
    berth = (1.0 if L1 > r["berth"] else 0.0) * p["b_extra"] * (0.5 + v)          # knock-on
    cong_proc = cong * 0.9 * r["proc"] * v ** 2
    t_start = t_arr + wait + berth
    wk = 1.0 if _dow(t_start) >= 5 else 0.0
    hol = 1.0 if int(t_start // 24) in HOLIDAYS else 0.0
    weekend = on("processing") * 0.35 * (r["proc"] + cong_proc) * wk
    holiday = on("processing") * 10.0 * hol
    pnoise = on("processing") * 2.0 * p["n2"]
    insp = on("inspection") * p["i2_flag"] * p["i2_size"]
    raw2 = L1 + wait + berth + cong_proc + weekend + holiday + pnoise + insp
    L2 = raw2 - e * min(r["s2"], max(raw2, 0.0))

    #################### S3: inland delivery
    appt = (1.0 if L2 > APPT_WINDOW_H else 0.0) * p["a_extra"]                      # knock-on
    inland = on("inland") * (p["in_flag"] * p["in_size"] + 1.5 * p["in_noise"])
    raw3 = L2 + inland + appt
    L3 = raw3 - e * min(r["s3"], max(raw3, 0.0))
    return dict(L0=L0, L1=L1, L2=L2, L3=L3, u_arr=uu, weekend_start=wk,
                port_wait = wait + berth,
                processing=r["proc"] + cong_proc +weekend + holiday + pnoise + insp,
                insp=p["i2_flag"] * on("inspection")
            )

def true_shapley(p: dict, r: dict, u: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Exact Shapley value of each delay source for the final delay. Value fn: v(S)=L3 with only S active."""
    n = len(SOURCES)
    v = {}
    for mask in range(1 << n):
        active = {SOURCES[i] for i in range(n) if mask >> i & 1}
        v[mask] = propagate(p, r, u, frozenset(set(SOURCES) - active))["L3"]
    phi = np.zeros(n)
    for i in range(n):
        for mask in range(1 << n):
            if mask >> i & 1:
                continue
            s = bin(mask).count("1")
            phi[i] += factorial(s) * factorial(n - s - 1) / factorial(n) * (v[mask | (1 << i)] - v[mask])

    return phi, v[0], v[(1 << n) - 1]


def generate_dataset(n: int = N_SHIPMENTS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    routes = build_routes()
    series = _port_series(rng)
    rows = []
    for _ in range(n):
        r = routes[int(rng.integers(len(routes)))]
        u = series[r["destination"]]
        dep_h = float(rng.integers(0, 270 * 24))
        dep_ts = BASE + pd.Timedelta(hours=dep_h)
        winter = 1.0 if dep_ts.month in (12, 1, 2) else 0.0
        oc = float(np.clip(rng.normal(r["oc_mean"], 0.15), 0.05, 0.95))
        p = dict(dep_h=dep_h, oc=oc,
                 o_flag=float(rng.random() < 0.25 + 0.5 * oc), o_size=float(rng.lognormal(np.log(6), 0.8)),
                 w=float(min(1.0, rng.beta(2, 5) * 1.3 * (1 + 0.4 * winter))), n1=float(rng.normal()),
                 qn=float(rng.lognormal(0, 0.3)), b_extra=float(rng.uniform(12, 30)),
                 i2_flag=float(rng.random() < 0.08), i2_size=float(rng.lognormal(np.log(20), 0.5)),
                 n2=float(rng.normal()), a_extra=float(rng.uniform(6, 14)),
                 in_flag=float(rng.random() < 0.30), in_size=float(rng.lognormal(np.log(5), 0.7)),
                 in_noise=float(rng.normal()))
        sim = propagate(p, r, u)
        phi, v0, vfull = true_shapley(p, r, u)

        if not np.isclose(phi.sum(), vfull - v0):
            raise AssertionError(
                f"Shapley efficiency failed for shipment "
                f"on route {r['route_id']}"
            )
        if not np.isclose(vfull, sim["L3"]):
            raise AssertionError(
                "Full-source Shapley outcome does not match simulation"
            )

        dep_s = dep_ts
        arr_s = dep_s + pd.Timedelta(hours=r["T1"])
        cus_s = arr_s + pd.Timedelta(hours=r["P"])
        del_s = cus_s + pd.Timedelta(hours=r["T3"])
        arr_a = arr_s + pd.Timedelta(hours=sim["L1"])
        cus_a = cus_s + pd.Timedelta(hours=sim["L2"])

        # missing intermediate data (non-random)
        p1 = 0.06 + (0.22 if r["destination"] in POOR_REPORTING else 0.0)
        p2 = 0.07 + 0.12 * sim["insp"] + 0.10 * sim["weekend_start"]
        miss1, miss2 = rng.random() < p1, rng.random() < p2

        row = dict(
            route_id=r["route_id"], origin=r["origin"], destination=r["destination"],
            dep_sched_ts=dep_s, port_arr_sched_ts=arr_s, customs_sched_ts=cus_s, delivery_sched_ts=del_s,
            dep_actual_ts=dep_s + pd.Timedelta(hours=sim["L0"]),
            port_arrival_actual_ts=pd.NaT if miss1 else arr_a,
            customs_cleared_actual_ts=pd.NaT if miss2 else cus_a,
            delivery_actual_ts=del_s + pd.Timedelta(hours=sim["L3"]),
            sched_transit_h=r["T1"], slack_transit_h=r["s1"], sched_port_h=r["P"], slack_port_h=r["s2"],
            sched_inland_h=r["T3"], slack_inland_h=r["s3"], proc_nominal_h=r["proc"],
            berth_window_h=r["berth"], appt_window_h=APPT_WINDOW_H, weather_exposure=r["exposure"],
            origin_congestion=float(np.clip(oc + rng.normal(0, 0.03), 0, 1)),
            weather_fcst=float(np.clip(p["w"] + rng.normal(0, 0.12), 0, 1)),
            weather_actual=float(np.clip(p["w"] + rng.normal(0, 0.03), 0, 1)),
            dest_congestion_dep=float(np.clip(u[int(dep_h // 24)] + rng.normal(0, 0.02), 0.3, 0.97)),
            dest_congestion_arr=float(np.clip(sim["u_arr"] + rng.normal(0, 0.02), 0.3, 0.97)),
            port_wait_h=np.nan if miss2 else sim["port_wait"],
            processing_h=np.nan if miss2 else sim["processing"],
            inspection_flag=np.nan if miss2 else sim["insp"],
            true_L0=sim["L0"], true_L1=sim["L1"], true_L2=sim["L2"], true_L3=sim["L3"],
            true_baseline=v0,
        )
        for s, ph in zip(SOURCES, phi):
            row[f"true_phi_{s}"] = ph
        for k in range(4):
            row[f"true_phi_stage{k}"] = sum(ph for s, ph in zip(SOURCES, phi) if SOURCE_STAGE[s] == k)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("dep_sched_ts").reset_index(drop=True)
    df.insert(0, "shipment_id", [f"SH{i:04d}" for i in range(len(df))])
    return df


def load_or_generate(n: int = N_SHIPMENTS, seed: int = SEED, force: bool = False) -> pd.DataFrame:
    path = DATA_DIR / f"shipments_n{n}_s{seed}.csv"
    ts_cols = ["dep_sched_ts", "port_arr_sched_ts", "customs_sched_ts", "delivery_sched_ts", "dep_actual_ts",
               "port_arrival_actual_ts", "customs_cleared_actual_ts", "delivery_actual_ts"]
    if path.exists() and not force:
        return pd.read_csv(path, parse_dates=ts_cols)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(n, seed)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    d = load_or_generate(force=True)
    print(d.shape)
    print(d["true_L3"].describe())


