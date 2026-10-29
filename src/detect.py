"""Run the trained model over a replayed CSV (or a live capture) and respond to attacks.
Also writes results/feed.csv and results/run_status.json, which the dashboard reads.

  python src/detect.py --csv data/raw/deauth/Deauth_25.csv --delay 0.3
  python src/detect.py --live wlan0mon --seconds 10     # own authorised lab network only"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from features import build_windows
from prevent import Responder
from sources import CSV_MAP, FRAME_COLS, LiveSniffSource

FEED_COLS = ["sec", "t0", "ta", "n", "rate_fps", "deauth_count",
             "mean_signal", "mgmt_frac", "pred"]


def load_csv(path):
    d = pd.read_csv(path, usecols=list(CSV_MAP), low_memory=False).rename(columns=CSV_MAP)
    d["file"] = os.path.basename(path)
    return d[FRAME_COLS]


def write_status(path, **kw):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(kw, fh)
    os.replace(tmp, path)  # atomic, so the dashboard never reads a half-written file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="replay one AWID3 CSV")
    ap.add_argument("--live", help="monitor-mode interface for live capture")
    ap.add_argument("--seconds", type=int, default=10, help="live capture length")
    ap.add_argument("--model", default="models/wips_rf.joblib")
    ap.add_argument("--delay", type=float, default=0.0, help="pause per second of capture")
    ap.add_argument("--protect", nargs="*", default=[], help="MACs that are never blocklisted")
    ap.add_argument("--enforce-iface", help="hostapd interface to enforce blocks on (own AP only)")
    ap.add_argument("--feed", default="results/feed.csv")
    ap.add_argument("--status", default="results/run_status.json")
    a = ap.parse_args()
    if bool(a.csv) == bool(a.live):
        ap.error("give exactly one of --csv or --live")

    os.makedirs(os.path.dirname(a.feed), exist_ok=True)
    source = a.csv or a.live
    write_status(a.status, state="loading", source=source, sec_done=0, sec_total=0)

    bundle = joblib.load(a.model)
    model, feats, window = bundle["model"], bundle["features"], bundle["window"]
    frames = load_csv(a.csv) if a.csv else LiveSniffSource(a.live, a.seconds).frames()
    w = build_windows(frames, window)
    if w.empty:
        write_status(a.status, state="done", source=source, sec_done=0, sec_total=0)
        print("No frames to analyse.")
        return
    w["sec"] = np.floor(w["t0"]).astype("int64")
    t_start = int(w["sec"].min())
    secs = sorted(w["sec"].unique())
    resp = Responder(protected=a.protect, enforce=bool(a.enforce_iface), iface=a.enforce_iface)

    pd.DataFrame(columns=FEED_COLS).to_csv(a.feed, index=False)  # fresh feed for this run
    print(f"Source: {source} | {len(frames)} frames -> {len(w)} windows of {window}s\n")
    done, alerts = [], 0
    for i, sec in enumerate(secs, 1):
        g = w[w["sec"] == sec].copy()
        g["pred"] = model.predict(g[feats])
        for _, r in g[g["pred"] == 1].iterrows():
            msg = resp.handle(r["ta"], r["t0"],
                              f"deauth_frames={int(r['deauth_count'])} rate={r['rate_fps']:.0f}fps")
            if msg:
                alerts += 1
                print("  ALERT", msg)
        print(f"t=+{sec - t_start:>3}s  windows={len(g):>4}  flagged={int(g['pred'].sum()):>3}")
        g.assign(sec=sec - t_start)[FEED_COLS].to_csv(a.feed, mode="a", header=False, index=False)
        write_status(a.status, state="running", source=source, sec_done=i, sec_total=len(secs))
        done.append(g)
        time.sleep(a.delay)

    write_status(a.status, state="done", source=source, sec_done=len(secs), sec_total=len(secs))
    res = pd.concat(done)
    print(f"\nAlerts raised: {alerts} | blocklist: {sorted(resp.blocked) or 'empty'}")
    if a.csv:
        cm = confusion_matrix(res["label"], res["pred"], labels=[0, 1])
        print(f"Confusion [[TN FP][FN TP]] on this file: {cm.tolist()}")


if __name__ == "__main__":
    main()
