"""Interactive demo:  streamlit run app/demo.py
Pick a shipment + checkpoint -> ETA with interval, delay risk, stage waterfall, SHAP drivers, what-if sliders.
The selected shipment is HELD OUT of training, so the demo shows a genuine out-of-sample prediction."""
import sys
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import CP_NAMES, DELAY_THRESHOLD_H
from src.explain import GROUP_LABELS, STAGE_NAMES, narrate, stage_attribution, what_if
from src.features import add_lateness
from src.generator import SOURCES, load_or_generate
from src.models import ETAPredictor

st.set_page_config(page_title="Shipment ETA & delay propagation", layout="wide")
st.title("Shipment ETA & delay propagation")


@st.cache_data
def data():
    return load_or_generate()


@st.cache_resource
def model(cp: int, exclude_id: str):
    df = data()
    return ETAPredictor(cp).fit(df[df.shipment_id != exclude_id])


df = data()
sid = st.sidebar.selectbox("Shipment", df.shipment_id.tolist(), index=min(25, len(df) - 1))
cp = st.sidebar.radio("Checkpoint (information available)", list(CP_NAMES), format_func=CP_NAMES.get, index=1)
row = df[df.shipment_id == sid]
r = row.iloc[0]
st.caption(f"{r.origin} → {r.destination} · route {r.route_id} · scheduled delivery {r.delivery_sched_ts:%d %b %Y %H:%M} UTC")

with st.spinner("Training on all other shipments..."):
    P = model(cp, sid)

st.sidebar.markdown("### What-if")
cong = st.sidebar.slider("Destination port congestion (utilisation)", 0.35, 0.95,
                         float(r.dest_congestion_arr if cp >= 1 else r.dest_congestion_dep), 0.01)
wx = st.sidebar.slider("Weather severity", 0.0, 1.0, float(r.weather_actual if cp >= 1 else r.weather_fcst), 0.01)

out = P.predict(row).iloc[0]
alt = what_if(P, row, congestion=cong, weather=wx).iloc[0]
eta = lambda h: r.delivery_sched_ts + pd.Timedelta(hours=float(h))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted ETA (UTC)", f"{eta(out.delay_chain):%d %b %H:%M}", f"{out.delay_chain:+.1f} h vs schedule", delta_color="inverse")
c2.metric("80% interval", f"{out.c10:+.0f} … {out.c90:+.0f} h", "chain Monte-Carlo", delta_color="off")
c3.metric(f"P(delay > {DELAY_THRESHOLD_H:.0f} h)", f"{out.p_delay_chain:.0%}", f"classifier {out.p_delay_clf:.0%}", delta_color="off")
c4.metric("What-if ETA shift", f"{alt.delay_chain - out.delay_chain:+.1f} h", f"{eta(alt.delay_chain):%d %b %H:%M}", delta_color="inverse")

A = stage_attribution(P, row).iloc[0]
left, right = st.columns(2)
with left:
    st.subheader("Where the delay comes from (by stage)")
    vals = [A[f"S{k}"] for k in range(4)] + [A["other"]]
    labels = [STAGE_NAMES[k] for k in range(4)] + ["baseline + propagation"]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.barh(labels[::-1], vals[::-1], color=["#ef4444" if v > 0 else "#10b981" for v in vals[::-1]])
    ax.axvline(0, color="k", lw=.6); ax.set_xlabel("hours")
    st.pyplot(fig)
    st.caption("Observed stage delay (already incurred) + model-attributed expected delay still to come.")
with right:
    st.subheader("Model drivers (SHAP, grouped)")
    g, _, _ = P.shap_groups(row)
    g = g.iloc[0].rename(index=GROUP_LABELS).sort_values()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.barh(g.index, g.values, color=["#ef4444" if v > 0 else "#10b981" for v in g.values]); ax.axvline(0, color="k", lw=.6)
    ax.set_xlabel("hours added to predicted remaining delay")
    st.pyplot(fig)

st.subheader("Plain-language explanation")
st.code(narrate(P, row, normal_congestion=float(df[df.destination == r.destination].dest_congestion_arr.median())))

with st.expander("Ground truth (synthetic data only): exact causal contribution of each delay source"):
    t = row[[f"true_phi_{s}" for s in SOURCES]].iloc[0]
    st.bar_chart(t.rename(index=lambda s: s.replace("true_phi_", "")))
    st.write(f"Actual final delay: **{add_lateness(row).L3.iloc[0]:+.1f} h**")
