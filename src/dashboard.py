"""WIPS dashboard.   Run from the project root:   streamlit run src/dashboard.py

It starts src/detect.py as a background process (replay mode) and reads the files
detect.py writes (results/feed.csv, run_status.json, alerts.log, blocklist.txt).
The same dashboard works for a live capture: run detect.py --live in a terminal."""
import glob
import json
import os
import subprocess
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

FEED = "results/feed.csv"
STATUS = "results/run_status.json"
ALERTS = "results/alerts.log"
BLOCK = "results/blocklist.txt"
CV = "results/cv_metrics.json"
EXP_NAMES = {"A_all_features": "A: all features",
             "B_no_deauth_count": "B: no deauth-count feature"}

st.set_page_config(page_title="WIPS Dashboard", page_icon="📡", layout="wide")


# ---------- helpers ----------
def read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def read_lines(path, n=None):
    try:
        with open(path) as fh:
            lines = [ln.rstrip("\n") for ln in fh if ln.strip() and not ln.startswith("#")]
    except OSError:
        return []
    return lines[-n:] if n else lines


def read_feed():
    try:
        d = pd.read_csv(FEED, on_bad_lines="skip")
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    for c in d.columns:
        if c != "ta":
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["sec", "ta"])


def stop_proc():
    p = st.session_state.get("proc")
    if p is not None and p.poll() is None:
        p.terminate()
    st.session_state["proc"] = None


# ---------- sidebar: replay controls ----------
with st.sidebar:
    st.title("📡 WIPS")
    st.caption("Wireless intrusion prevention: deauthentication-flood detection")
    st.subheader("Replay control")
    files = sorted(glob.glob("data/raw/*/*.csv"))
    path = None
    if files:
        default = next((i for i, f in enumerate(files) if f.endswith("Deauth_25.csv")), 0)
        path = st.selectbox("Capture file", files, index=default,
                            format_func=lambda p: f"{os.path.basename(p)}  ({p.split('/')[2]})")
    else:
        st.warning("No CSVs found in data/raw/. Run the dashboard from the project root.")
    delay = st.slider("Seconds per second of capture", 0.0, 1.0, 0.3, 0.1,
                      help="0 = as fast as possible. 1 = real time.")
    c1, c2 = st.columns(2)
    if c1.button("▶ Start", type="primary", disabled=path is None, width="stretch"):
        stop_proc()
        os.makedirs("results", exist_ok=True)
        for p in (FEED, STATUS, ALERTS, BLOCK):   # clean slate for the demo
            if os.path.exists(p):
                os.remove(p)
        st.session_state["proc"] = subprocess.Popen(
            [sys.executable, "src/detect.py", "--csv", path, "--delay", str(delay)],
            stdout=open("results/detect_stdout.log", "w"), stderr=subprocess.STDOUT)
    if c2.button("■ Stop", width="stretch"):
        stop_proc()
    st.divider()
    st.caption("Replay mode. For a live capture on your own authorised lab network, run "
               "`python src/detect.py --live wlan0mon` in a terminal. This dashboard then "
               "shows it the same way.")


# ---------- live view (refreshes every second) ----------
@st.fragment(run_every=1)
def live_view():
    status, feed = read_json(STATUS), read_feed()

    if feed.empty:
        if status.get("state") == "loading":
            st.info("Loading the capture and building features…")
        else:
            st.info("No data yet. Pick a capture file in the sidebar and press **Start**.")
        return

    latest = feed["sec"].max()
    recent = feed[feed["sec"] >= latest - 2]
    attackers = sorted(recent.loc[recent["pred"] == 1, "ta"].unique())
    if attackers:
        st.error("🚨 **UNDER ATTACK: deauthentication flood detected.** "
                 f"Source MAC(s) in the attack frames: {', '.join(attackers)}")
    else:
        st.success("✅ **Network normal.** No deauthentication flood in the last 3 seconds.")

    total = status.get("sec_total", 0)
    if total:
        st.progress(min(status.get("sec_done", 0) / total, 1.0),
                    text=f"{status.get('state', '').capitalize()}: "
                         f"{status.get('sec_done', 0)} of {total} seconds of capture analysed")

    n_alerts, blocked = len(read_lines(ALERTS)), read_lines(BLOCK)
    k = st.columns(5)
    k[0].metric("Frames analysed", f"{int(feed['n'].sum()):,}")
    k[1].metric("Devices seen", feed["ta"].nunique())
    k[2].metric("Deauth frames", f"{int(feed['deauth_count'].sum()):,}")
    k[3].metric("Alerts raised", n_alerts)
    k[4].metric("MACs blocklisted", len(blocked))

    per_sec = (feed.groupby("sec")
                   .agg(frames=("n", "sum"), deauth=("deauth_count", "sum"),
                        flagged=("pred", "sum"), signal=("mean_signal", "mean"))
                   .reset_index())
    long = per_sec.melt(id_vars="sec", value_vars=["frames", "deauth"],
                        var_name="series", value_name="count")
    long["series"] = long["series"].map({"frames": "All frames", "deauth": "Deauth frames"})
    line = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("sec:Q", title="Seconds since start"),
        y=alt.Y("count:Q", title="Frames per second"),
        color=alt.Color("series:N", title=None,
                        scale=alt.Scale(domain=["All frames", "Deauth frames"],
                                        range=["#4c78a8", "#e45756"])))
    att = per_sec[per_sec["flagged"] > 0].assign(sec2=lambda d: d["sec"] + 1)
    chart = line
    if len(att):
        chart = alt.Chart(att).mark_rect(color="#e45756", opacity=0.15).encode(
            x="sec:Q", x2="sec2:Q") + line

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Traffic per second")
        st.caption("Red shading = seconds when the model flagged an attack.")
        st.altair_chart(chart, width="stretch")
    with right:
        st.subheader("Average signal strength (dBm)")
        st.caption("Closer to 0 = stronger signal.")
        st.line_chart(per_sec.set_index("sec")["signal"])

    st.subheader("Devices on the air")
    dev = (feed.groupby("ta")
               .agg(frames=("n", "sum"), deauth_frames=("deauth_count", "sum"),
                    flagged_windows=("pred", "sum"), avg_signal_dbm=("mean_signal", "mean"),
                    peak_rate_fps=("rate_fps", "max"))
               .reset_index()
               .sort_values(["flagged_windows", "deauth_frames", "frames"], ascending=False)
               .head(10))
    dev["status"] = np.where(dev["flagged_windows"] > 0, "⚠ flagged", "ok")
    dev["avg_signal_dbm"] = dev["avg_signal_dbm"].round(1)
    dev = dev.rename(columns={"ta": "device (transmitter MAC)"})
    st.dataframe(dev, hide_index=True, width="stretch")
    st.caption("Attackers usually spoof the MAC of your access point or a client, so a flagged "
               "MAC is the address the frames *claimed* to come from.")

    a_col, b_col = st.columns([3, 2])
    with a_col:
        st.subheader("Alerts")
        alerts = read_lines(ALERTS, 12)
        if alerts:
            st.code("\n".join(reversed(alerts)), language=None)
        else:
            st.caption("No alerts yet.")
    with b_col:
        st.subheader("Blocklist (prevention)")
        if blocked:
            st.code("\n".join(b.split(" #")[0] for b in blocked), language=None)
        else:
            st.caption("Empty. MACs are added automatically when an attack is detected.")


tab_live, tab_model, tab_how = st.tabs(["Live monitor", "Model performance", "How it works"])

with tab_live:
    live_view()

with tab_model:
    st.subheader("How well does the detector work?")
    cv = read_json(CV)
    if isinstance(cv, list) and cv:
        t = pd.DataFrame(cv)
        t["experiment"] = t["experiment"].map(EXP_NAMES).fillna(t["experiment"])
        cols = ["experiment", "model", "precision_mean", "recall_mean", "f1_mean", "accuracy_mean"]
        st.dataframe(t[cols].round(3), hide_index=True, width="stretch")
        st.caption("Grouped 4-fold cross-validation, repeated over 5 random splits. About 0.3% of "
                   "windows are attacks, so accuracy is always high; precision, recall and F1 "
                   "are the numbers that matter.")
        imgs = sorted(glob.glob("results/cv_cm_*.png"))
        for i in range(0, len(imgs), 3):
            row = st.columns(3)
            for col, img in zip(row, imgs[i:i + 3]):
                col.image(img, caption=os.path.basename(img)[6:-4].replace("_", " "))
    else:
        st.info("No results yet. Run `python src/crossval.py` first.")
    st.warning("Limitations: only one 14-second deauth attack in the dataset, and its normal traffic "
               "has no legitimate deauth bursts, so real-world false-alarm rates are untested.")

with tab_how:
    st.markdown("""
**1. Read traffic.** Frames come from a replayed AWID3 capture (or a live monitor-mode capture).

**2. Summarise.** For every device (transmitter MAC) and every 0.25 s window we compute the frame
rate, the number and share of deauthentication frames, management-frame share, signal strength
and frame length.

**3. Classify.** A Random Forest labels each window *normal* or *deauth flood*.

**4. Respond.** On an attack the system raises an alert, logs it, and adds the offending MAC to a
blocklist (the WIPS action).

A deauthentication flood forces clients off a Wi-Fi network by sending forged "disconnect" frames
in large numbers. The tell-tale sign is a sudden burst of deauth frames, often 150+ per second.
""")
