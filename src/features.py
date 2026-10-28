"""Turn a frame table into one row per (device, 0.25 s window)."""
import numpy as np
import pandas as pd

WINDOW = 0.25  # seconds

FEATURES_ALL = ["rate_fps", "deauth_count", "deauth_frac", "mgmt_frac",
                "mean_signal", "std_signal", "mean_len", "n_unique_ra"]
# Experiment B: the model may not look at deauth counts directly.
FEATURES_NO_DEAUTH = [f for f in FEATURES_ALL if not f.startswith("deauth")]


def build_windows(frames, window=WINDOW):
    f = frames.dropna(subset=["ta"]).copy()   # control frames have no transmitter
    for c in ("ts", "ftype", "subtype", "signal", "length"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f["win"] = np.floor(f["ts"] / window).astype("int64")
    f["is_deauth"] = ((f["ftype"] == 0) & (f["subtype"] == 12)).astype(int)
    f["is_mgmt"] = (f["ftype"] == 0).astype(int)
    f["is_attack"] = (f["label"] == "Deauth").astype(int)

    w = (f.groupby(["file", "win", "ta"], observed=True)
          .agg(n=("ts", "size"), deauth_count=("is_deauth", "sum"),
               mgmt_frac=("is_mgmt", "mean"), mean_signal=("signal", "mean"),
               std_signal=("signal", "std"), mean_len=("length", "mean"),
               n_unique_ra=("ra", "nunique"), label=("is_attack", "max"),
               t0=("ts", "min"))
          .reset_index())

    w["rate_fps"] = w["n"] / window
    w["deauth_frac"] = w["deauth_count"] / w["n"]
    w["std_signal"] = w["std_signal"].fillna(0)
    w["mean_signal"] = w["mean_signal"].fillna(-100)
    w["mean_len"] = w["mean_len"].fillna(0)
    return w
