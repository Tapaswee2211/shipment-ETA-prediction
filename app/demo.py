import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import CP_NAMES, DELAY_THRESHOLD_H, OUT_DIR
from src.explain import GROUP_LABELS, STAGE_NAMES, explain_text, stage_breakdown
from src.generator import SOURCES, SOURCE_STAGE, load_or_generate
from src.inputs import (CONGESTION_RANGE, ORIGINS, NewShipment, as_of, build_row, destinations_for, holdout_split,
                        lane_properties, normalize_id, typical_conditions, validate_new_shipment)
from src.models import ETAPredictor
from src.scenario import apply_whatif, available, sweep

H = pd.Timedelta(hours=1)
INK, TEAL, SLATE, RED, GREEN = "#0f172a", "#0f766e", "#64748b", "#dc2626", "#16a34a"
STATUS_COLOR = {"observed": INK, "estimated": "#94a3b8", "expected": TEAL}
THR = DELAY_THRESHOLD_H

st.set_page_config(page_title="Shipment ETA Intelligence", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2.6rem;max-width:1180px}
.app-title{font-size:1.75rem;font-weight:700;color:#0f172a;margin:0;letter-spacing:-.01em}
.app-sub{color:#64748b;font-size:.92rem;margin:.15rem 0 1rem}
.strip{display:flex;gap:30px;flex-wrap:wrap;padding:6px 0 14px;border-bottom:1px solid #e2e8f0;margin-bottom:18px}
.strip .k{font-size:.7rem;letter-spacing:.07em;text-transform:uppercase;color:#64748b}
.strip .v{font-size:.95rem;font-weight:600;color:#0f172a}
.hero{background:#0f172a;border-radius:14px;padding:24px 28px;color:#f8fafc}
.hero .label{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8}
.hero .eta{font-size:2.3rem;font-weight:700;line-height:1.2;margin:.3rem 0 .1rem}
.hero .eta span{font-size:1rem;font-weight:500;color:#94a3b8}
.hero .delta{font-size:1.2rem;font-weight:600}
.hero .late{color:#fca5a5}.hero .early{color:#86efac}.hero .ontime{color:#e2e8f0}
.hero .sub{color:#94a3b8;font-size:.85rem;margin-top:.55rem}
.card{border:1px solid #e2e8f0;border-radius:12px;padding:13px 18px;background:#fff;margin-bottom:12px}
.card .label{font-size:.7rem;letter-spacing:.07em;text-transform:uppercase;color:#64748b}
.card .value{font-size:1.4rem;font-weight:700;color:#0f172a;line-height:1.3}
.card .sub{font-size:.8rem;color:#64748b}
.note{font-size:.82rem;color:#64748b}
h3{font-size:1.1rem!important;margin-top:1.4rem!important}
</style>""", unsafe_allow_html=True)

_NEW_API = "width" in inspect.signature(st.plotly_chart).parameters


def show(fig, key):
    fig.update_layout(template="simple_white", font=dict(size=12, color=INK), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, key=key, **({"width": "stretch"} if _NEW_API else {"use_container_width": True}))


def table(data, hide_index=False):
    st.dataframe(data, hide_index=hide_index, **({"width": "stretch"} if _NEW_API else {"use_container_width": True}))


def fmt_ts(ts, year=False):
    return f"{ts:%a %d %b %Y, %H:%M}" if year else f"{ts:%a %d %b, %H:%M}"


def fmt_delay(h):
    if abs(h) < 0.5:
        return "on schedule", "ontime"
    return (f"{abs(h):.1f} h late", "late") if h > 0 else (f"{abs(h):.1f} h early", "early")


# data & models : cached
@st.cache_data(show_spinner=False)
def load_data():
    return load_or_generate()


@st.cache_resource(show_spinner="Training the checkpoint model (a few seconds)…", max_entries=12)
def get_model(cp: int, exclude_id: str):
    """exclude_id='' trains on every shipment; otherwise that shipment is removed from the training data."""
    df = load_data()
    return ETAPredictor(cp).fit(df if not exclude_id else holdout_split(df, exclude_id)[0])


def load_summary():
    p = OUT_DIR / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


df = load_data()
st.markdown('<div class="app-title">Shipment ETA Intelligence</div>'
            '<div class="app-sub">Estimated arrival, delay risk and where the delay comes from · synthetic shipment data · '
            'all times UTC</div>', unsafe_allow_html=True)

# sidebar: inputs
st.sidebar.markdown("## Shipment")
mode = st.sidebar.radio("Input mode", ["Existing shipment", "New shipment"], key="mode")
cp = st.sidebar.selectbox("Prediction checkpoint", [0, 1, 2], index=1, format_func=lambda c: CP_NAMES[c], key="cp",
                          help="What is known when the estimate is made: just departed, arrived at port, or customs cleared.")
full_row = None
warnings_ui = []

if mode == "Existing shipment":
    sid_in = st.sidebar.text_input("Shipment ID", "SH0100", key="sid")
    ids = df["shipment_id"]
    st.sidebar.caption(f"Valid IDs: {ids.iloc[0]} – {ids.iloc[-1]}. The selected shipment is excluded from the "
                       "model's training data.")
    if normalize_id(sid_in) not in set(ids.str.upper()):
        st.error(f"Unknown shipment ID '{sid_in}'. Enter an ID between {ids.iloc[0]} and {ids.iloc[-1]}.")
        st.stop()
    _, full_row = holdout_split(df, sid_in)
    sid = full_row["shipment_id"].iloc[0]
    row = as_of(full_row, cp)                     # the model only sees what was known at the checkpoint
    P = get_model(cp, sid)
    ctx = f"{sid}_{cp}"
else:
    st.sidebar.markdown("#### Shipment")
    ref = st.sidebar.text_input("Reference ID", "NEW-001", key="ref")
    origin = st.sidebar.selectbox("Origin", ORIGINS, key="origin")
    dest = st.sidebar.selectbox("Destination", destinations_for(origin), key="dest")
    lane = lane_properties(origin, dest)
    st.sidebar.caption(f"Lane {lane['route_id']} · scheduled transit {lane['sched_transit_h'] / 24:.0f} d · port dwell "
                       f"{lane['sched_port_h']:.0f} h · inland {lane['sched_inland_h']:.0f} h (from the route table)")
    dcol, tcol = st.sidebar.columns(2)
    dep_date = dcol.date_input("Departure date", df["dep_sched_ts"].median().date(), key="dep_date")
    dep_time = tcol.time_input("Time (UTC)", pd.Timestamp("08:00").time(), key="dep_time")
    typ, lk = typical_conditions(df, origin, dest), f"{origin}_{dest}"
    lo_c, hi_c = CONGESTION_RANGE

    st.sidebar.markdown("#### Known at departure")
    L0 = st.sidebar.number_input("Delay at departure (h; negative = early)", -6.0, 72.0, 0.0, 0.5, key="L0")
    oc = st.sidebar.slider("Origin port congestion", 0.0, 1.0, round(typ["origin_congestion"], 2), 0.01, key=f"oc_{lk}")
    wf = st.sidebar.slider("Weather outlook (0 calm – 1 severe)", 0.0, 1.0, round(typ["weather"], 2), 0.01, key=f"wf_{lk}")
    cd = st.sidebar.slider("Destination congestion forecast", lo_c, hi_c, float(np.clip(round(typ["dest_congestion"], 2), lo_c, hi_c)),
                           0.01, key=f"cd_{lk}", help="Port utilisation index: 0.30 quiet … 0.97 saturated")
    kw = {}
    if cp >= 1:
        st.sidebar.markdown("#### Observed at port arrival")
        kw["L1"] = st.sidebar.number_input("Delay at port arrival (h)", -24.0, 150.0, 3.0, 0.5, key="L1")
        kw["weather_actual"] = st.sidebar.slider("Weather encountered", 0.0, 1.0, float(wf), 0.01, key=f"wa_{lk}")
        kw["dest_congestion_arr"] = st.sidebar.slider("Destination congestion at arrival", lo_c, hi_c, float(cd), 0.01, key=f"ca_{lk}")
    if cp >= 2:
        st.sidebar.markdown("#### Observed at customs clearance")
        l2_default = round(3.0 + 8.0 + lane["proc_nominal_h"] + 0.3 * lane["slack_port_h"] - lane["sched_port_h"], 1)
        kw["L2"] = st.sidebar.number_input("Delay at customs clearance (h)", -24.0, 200.0, l2_default, 0.5, key=f"L2_{lk}")
        kw["port_wait_h"] = st.sidebar.number_input("Queue wait (h)", 0.0, 200.0, 8.0, 0.5, key="wait")
        kw["processing_h"] = st.sidebar.number_input("Processing time (h)", 0.0, 200.0, float(lane["proc_nominal_h"]), 0.5, key=f"proc_{lk}")
        kw["inspection"] = st.sidebar.checkbox("Container inspected", False, key="insp")
    st.sidebar.caption("Cargo attributes (weight, container type, carrier) are not features of this model.")

    inp = NewShipment(ref=ref, origin=origin, destination=dest, dep_sched=pd.Timestamp.combine(dep_date, dep_time),
                      cp=cp, L0=L0, origin_congestion=oc, weather_fcst=wf, dest_congestion_dep=cd, **kw)
    result = validate_new_shipment(inp, df["shipment_id"], (df["dep_sched_ts"].min(), df["dep_sched_ts"].max()))
    if result.errors:
        for e in result.errors:
            st.error(e)
        st.stop()
    warnings_ui = result.warnings
    row = build_row(inp, df)
    P = get_model(cp, "")
    ctx = f"{ref}_{lane['route_id']}_{cp}"

#  prediction
out = P.predict(row).iloc[0]
sched = row["delivery_sched_ts"].iloc[0]
med, lo, hi = float(out["delay_chain_median"]), float(out["c10"]), float(out["c90"])
eta = sched + med * H                              # delay already contains the observed delay: add it once
bd = stage_breakdown(out, cp)
r0 = row.iloc[0]

for w in warnings_ui:
    st.warning(w)

observed = [f"departed {out['chain_L0']:+.1f} h"]
if cp >= 1:
    observed.append(f"port arrival {out['chain_L1']:+.1f} h" + (" (estimated)" if out["L1_imputed"] else ""))
if cp >= 2:
    observed.append(f"customs {out['chain_L2']:+.1f} h" + (" (estimated)" if out["L2_imputed"] else ""))
st.markdown(f"""<div class="strip">
<div><div class="k">Shipment</div><div class="v">{r0['shipment_id']}</div></div>
<div><div class="k">Lane</div><div class="v">{r0['origin']} → {r0['destination']}</div></div>
<div><div class="k">Scheduled departure</div><div class="v">{fmt_ts(r0['dep_sched_ts'])}</div></div>
<div><div class="k">Scheduled delivery</div><div class="v">{fmt_ts(sched)}</div></div>
<div><div class="k">Checkpoint</div><div class="v">{CP_NAMES[cp]}</div></div>
<div><div class="k">Observed so far (vs schedule)</div><div class="v">{' · '.join(observed)}</div></div>
</div>""", unsafe_allow_html=True)

# hero
text, cls = fmt_delay(med)
left, right = st.columns([3, 2], gap="medium")
left.markdown(f"""<div class="hero"><div class="label">Predicted arrival · median estimate</div>
<div class="eta">{fmt_ts(eta, True)} <span>UTC</span></div>
<div class="delta {cls}">{text} vs scheduled delivery</div>
<div class="sub">Scheduled: {fmt_ts(sched, True)} UTC</div></div>""", unsafe_allow_html=True)
right.markdown(f"""<div class="card"><div class="label">80% prediction range</div>
<div class="value">{fmt_ts(sched + lo * H)} → {fmt_ts(sched + hi * H)}</div>
<div class="sub">{lo:+.1f} h to {hi:+.1f} h vs schedule · 10th–90th percentile of the simulation</div></div>
<div class="card"><div class="label">Probability of arriving more than {THR:g} h late</div>
<div class="value">{out['p_delay_chain']:.0%}</div>
<div class="sub">share of simulated outcomes with final delay &gt; {THR:g} h</div></div>""", unsafe_allow_html=True)

strip = go.Figure()
strip.add_trace(go.Scatter(x=[lo, hi], y=[0, 0], mode="lines", name="80% range", line=dict(width=28, color="rgba(15,118,110,.25)"),
                           hovertemplate="80% range: %{x:.1f} h<extra></extra>"))
strip.add_trace(go.Scatter(x=[med], y=[0], mode="markers", marker=dict(size=15, color=TEAL), name="median",
                           hovertemplate="median %{x:.1f} h<extra></extra>"))
strip.add_trace(go.Scatter(x=[float(out["L_k"])], y=[0], mode="markers", name="delay so far",
                           marker=dict(size=11, color=INK, symbol="diamond"), hovertemplate="delay so far %{x:.1f} h<extra></extra>"))
strip.add_vline(x=0, line=dict(color=SLATE, width=1.5, dash="dash"), annotation_text="scheduled", annotation_position="top")
strip.add_vline(x=THR, line=dict(color=RED, width=1.5, dash="dash"), annotation_text=f"{THR:g} h late", annotation_position="top")
xs = [lo, hi, med, float(out["L_k"]), 0.0, THR]
pad = 0.08 * (max(xs) - min(xs) + 1)
strip.update_xaxes(title="final delay vs scheduled delivery (h)", range=[min(xs) - pad, max(xs) + pad])
strip.update_yaxes(visible=False, range=[-0.6, 0.6])
strip.update_layout(height=150, margin=dict(l=0, r=0, t=26, b=0), legend=dict(orientation="h", y=-0.6))
show(strip, "strip")

if full_row is not None and st.checkbox("Reveal the actual outcome (synthetic data)", False, key="reveal"):
    act = full_row["delivery_actual_ts"].iloc[0]
    err = (eta - act) / H
    inside = sched + lo * H <= act <= sched + hi * H
    st.info(f"Actual arrival: **{fmt_ts(act, True)} UTC** ({(act - sched) / H:+.1f} h vs schedule). "
            f"The median estimate was {abs(err):.1f} h {'earlier' if err < 0 else 'later'} than the actual arrival, which "
            f"{'falls inside' if inside else 'falls outside'} the 80% range.")

# why
st.markdown("### Why this estimate")
c1, c2 = st.columns([3, 2], gap="large")
with c1:
    fig = go.Figure()
    for status, label in [("observed", "observed"), ("estimated", "estimated (timestamp missing)"), ("expected", "expected (still to come)")]:
        sub = bd[bd.status == status]
        if len(sub):
            fig.add_trace(go.Bar(x=sub.delta_h, y=sub.stage, orientation="h", name=label, marker_color=STATUS_COLOR[status],
                                 text=[f"{v:+.1f} h" for v in sub.delta_h], textposition="outside", cliponaxis=False))
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=list(bd.stage))
    fig.update_xaxes(title="hours added during the stage (negative = schedule buffer recovered time)", zeroline=True)
    fig.update_layout(height=280, margin=dict(l=0, r=40, t=0, b=0), legend=dict(orientation="h", y=-0.35), bargap=0.35)
    show(fig, "breakdown")
    st.markdown(f'<div class="note">Stage values add up to the model\'s <b>average</b> delay, {bd.delta_h.sum():+.1f} h '
                f'(= {sched:%d %b} schedule + {bd.delta_h.sum():+.1f} h). The headline uses the <b>median</b> ({med:+.1f} h): '
                'delays are right-skewed, so the average is higher. Observed stages are measured; expected stages are '
                'model averages.</div>', unsafe_allow_html=True)

# what-if inputs are needed for the explanation's counterfactual, so compute them first
fields = available(cp)
typ_now = typical_conditions(df, r0["origin"], r0["destination"])
cf = None
if "congestion" in fields:
    cur_c = float(r0[fields["congestion"]])
    if cur_c - typ_now["dest_congestion"] > 0.03:
        d_cf = float(P.predict(apply_whatif(row, cp, congestion=typ_now["dest_congestion"]))["delay_chain_median"].iloc[0]) - med
        if abs(d_cf) >= 0.5:
            cf = dict(label=f"destination congestion at its typical level ({typ_now['dest_congestion']:.2f} instead of {cur_c:.2f})",
                      delta_h=d_cf)
with c2:
    st.markdown("**In plain language**")
    st.markdown("\n".join(f"- {t}" for t in explain_text(out, bd, cp, THR, cf)))

#  what-if
st.markdown("### What-if: weather and port congestion")
if not fields:
    st.info("At this checkpoint the port stage is already complete, so weather and port congestion can no longer change "
            "the remaining (inland) stage in this model. Choose an earlier checkpoint to explore them.")
else:
    w1, w2 = st.columns(2)
    scen_c = scen_w = None
    cur_c = float(r0[fields["congestion"]])
    scen_c = w1.slider("Destination port congestion", CONGESTION_RANGE[0], CONGESTION_RANGE[1],
                       float(np.clip(cur_c, *CONGESTION_RANGE)), 0.01, key=f"wi_c_{ctx}_{cur_c:.2f}")
    if "weather" in fields:
        cur_w = float(r0[fields["weather"]])
        scen_w = w2.slider("Weather severity outlook", 0.0, 1.0, float(np.clip(cur_w, 0, 1)), 0.01, key=f"wi_w_{ctx}_{cur_w:.2f}")
    else:
        w2.markdown('<div class="note" style="padding-top:1.9rem">Weather is not adjustable here: the ocean transit is already complete.</div>',
                    unsafe_allow_html=True)
    out_w = P.predict(apply_whatif(row, cp, congestion=scen_c, weather=scen_w)).iloc[0]
    m_w = float(out_w["delay_chain_median"])
    cmp_df = pd.DataFrame({
        "": ["Median arrival (UTC)", "Delay vs schedule", f"P(> {THR:g} h late)", "80% range (h vs schedule)"],
        "Current inputs": [fmt_ts(eta), f"{med:+.1f} h", f"{out['p_delay_chain']:.0%}", f"{lo:+.1f} … {hi:+.1f}"],
        "Scenario": [fmt_ts(sched + m_w * H), f"{m_w:+.1f} h ({m_w - med:+.1f})", f"{out_w['p_delay_chain']:.0%}",
                     f"{out_w['c10']:+.1f} … {out_w['c90']:+.1f}"]})
    t_col, g_col = st.columns([2, 3], gap="large")
    with t_col:
        table(cmp_df, hide_index=True)
    grid = np.linspace(CONGESTION_RANGE[0], 0.95, 13)
    sw = sweep(P, apply_whatif(row, cp, weather=scen_w) if scen_w is not None else row, "congestion", grid)
    cf_fig = go.Figure()
    cf_fig.add_trace(go.Scatter(x=np.r_[sw.x, sw.x[::-1]], y=np.r_[sw.hi, sw.lo[::-1]], fill="toself",
                                fillcolor="rgba(15,118,110,.15)", line=dict(width=0), name="80% range", hoverinfo="skip"))
    cf_fig.add_trace(go.Scatter(x=sw.x, y=sw["median"], mode="lines", line=dict(color=TEAL, width=3), name="median delay"))
    cf_fig.add_trace(go.Scatter(x=[scen_c], y=[m_w], mode="markers", marker=dict(size=11, color=INK), name="scenario"))
    cf_fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=-0.35),
                         xaxis_title="destination congestion", yaxis_title="final delay (h)")
    g_col.markdown('<div class="note">Median delay as destination congestion varies (weather and everything else held at the '
                   'scenario values).</div>', unsafe_allow_html=True)
    with g_col:
        show(cf_fig, "sens")

#  diagnostics
st.markdown("### ")
with st.expander("Technical diagnostics"):
    t1, t2, t3, t4, t5 = st.tabs(["Outputs & definitions", "Model input vector", "SHAP (GBM comparator)",
                                  "Synthetic ground truth", "Measured evaluation"])
    with t1:
        st.markdown(f"""
**All delays are hours versus the scheduled delivery and already include the delay observed so far (`L_k`).**
Predicted arrival = scheduled delivery + delay (the observed delay is added once). The models are trained on the
*remaining* delay `R_k = L_3 − L_k`, and `delay = L_k + R̂_k`.

- **Headline model (chain):** predicts each stage's change in lateness from the state at the start of that stage and
  simulates the rest of the journey 200 times, drawing noise from out-of-fold residuals. Median = 50th percentile;
  the 80% range = 10th–90th percentile; P(late) = share of draws with delay > {THR:g} h.
- **Comparators:** gradient-boosted trees (GBM), ridge regression, "carry the current delay forward", "trust the schedule".
""")
        table(pd.DataFrame({"output": ["delay_chain_median (headline)", "delay_chain (mean of simulation)", "c10 / c90",
                                              "p_delay_chain", "p_delay_clf (separate classifier, cross-check)",
                                              "delay_gbm", "delay_ridge", "delay_carry (= L_k)"],
                                   "value": [med, out.delay_chain, f"{lo:.1f} / {hi:.1f}", f"{out.p_delay_chain:.3f}",
                                             f"{out.p_delay_clf:.3f}", out.delay_gbm, out.delay_ridge, out.delay_carry]}).astype(str), hide_index=True)
    with t2:
        X = P.feature_row(row)
        st.markdown(f"Exact feature vector passed to the GBM / ridge models at {CP_NAMES[cp]} ({X.shape[1]} features). "
                    "Blank = missing; unavailable-at-checkpoint fields are never filled with placeholders.")
        table(X.T.rename(columns={X.index[0]: "value"}).astype(str))
    with t3:
        sg, base, _ = P.shap_groups(row)
        sg = sg.iloc[0].rename(index=GROUP_LABELS).sort_values()
        st.markdown(f"SHAP values of the **GBM comparator** for its *remaining-delay* estimate (hours): "
                    f"GBM delay {out.delay_gbm:+.1f} h = delay so far {out.L_k:+.1f} + baseline {base:+.1f} + Σ SHAP "
                    f"{sg.sum():+.1f}. They explain the GBM, not the headline chain model, and describe model "
                    "behaviour, not causes.")
        bar = go.Figure(go.Bar(x=sg.values, y=sg.index, orientation="h", marker_color=[RED if v > 0 else GREEN for v in sg.values]))
        bar.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="hours")
        show(bar, "shap")
    with t4:
        if full_row is None:
            st.info("Available only for existing synthetic shipments (a new shipment has no simulated outcome).")
        else:
            phi = full_row[[f"true_phi_{s}" for s in SOURCES]].iloc[0]
            st.markdown(f"The data generator's own counterfactual attribution of the *realised* final delay "
                        f"({full_row['true_L3'].iloc[0]:+.1f} h): exact Shapley values over the generator's delay sources. "
                        "It shows what the simulator did for this synthetic shipment; **it is not evidence of real-world "
                        "causality**, and it uses the realised outcome, which the model cannot see.")
            tr = pd.Series(phi.values, index=[f"{s.replace('_', ' ')} (stage {SOURCE_STAGE[s]}: {STAGE_NAMES[SOURCE_STAGE[s]]})" for s in SOURCES])
            tb = go.Figure(go.Bar(x=tr.values, y=tr.index, orientation="h", marker_color=[RED if v > 0 else GREEN for v in tr.values]))
            tb.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="hours")
            tb.update_yaxes(autorange="reversed")
            show(tb, "truth")
    with t5:
        s = load_summary()
        if s is None:
            st.info("Run `python -m src.pipeline` to generate `outputs/summary.json` with the cross-validation results.")
        else:
            st.markdown(f"Out-of-sample results from `python -m src.pipeline` (n={s['n']}, seed {s['seed']}, 5-fold CV × "
                        f"{s['cv_repeats']}). MAE in hours of final-arrival error.")
            order = ["sched", "carry", "ridge", "gbm", "chain", "chain_median"]
            table(pd.DataFrame({CP_NAMES[int(cp_)]: {m: s["mae"][cp_].get(m) for m in order} for cp_ in s["mae"]})
                         .rename(index={"sched": "trust the schedule", "carry": "carry current delay", "chain": "chain (mean)",
                                        "chain_median": "chain (median) = headline"}))
            st.markdown("80% range empirical coverage: " + "; ".join(
                f"{CP_NAMES[int(c)]}: {v['chain_monte_carlo']:.0%}" for c, v in s["interval_coverage_80"].items())
                + f" (target 80%). Brier score of P(late > {s['delay_threshold_h']:g} h), chain: "
                + "; ".join(f"{CP_NAMES[int(c)]}: {v['chain_monte_carlo']:.3f}" for c, v in s["brier"].items()) + ".")
