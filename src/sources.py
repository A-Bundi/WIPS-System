"""Packet sources. Each source returns the same frame table, so the
feature extractor, model and detector never care whether frames come
from a CSV replay or a live capture."""
import glob
import os

import pandas as pd

FRAME_COLS = ["file", "ts", "ftype", "subtype", "ta", "ra", "signal", "length", "label"]

CSV_MAP = {
    "frame.time_epoch": "ts",
    "wlan.fc.type": "ftype",
    "wlan.fc.subtype": "subtype",
    "wlan.ta": "ta",
    "wlan.ra": "ra",
    "wlan_radio.signal_dbm": "signal",
    "frame.len": "length",
    "Label": "label",
}

# Their "Normal" rows contain real deauth floods (found during data inspection).
EXCLUDE_FILES = {"RogueAP_31.csv", "RogueAP_33.csv"}


class CsvReplaySource:
    """Replays AWID3 CSV captures as a frame table."""

    def __init__(self, root="data/raw", keep_labels=("Normal", "Deauth")):
        self.root = root
        self.keep_labels = set(keep_labels)

    def files(self):
        paths = sorted(glob.glob(os.path.join(self.root, "*", "*.csv")))
        return [p for p in paths if os.path.basename(p) not in EXCLUDE_FILES]

    def frames(self):
        parts = []
        paths = self.files()
        for i, p in enumerate(paths, 1):
            d = pd.read_csv(p, usecols=list(CSV_MAP), low_memory=False)
            d = d.rename(columns=CSV_MAP)
            d = d[d["label"].isin(self.keep_labels)].copy()
            d["file"] = os.path.basename(p)
            parts.append(d[FRAME_COLS])
            print(f"  loaded {i}/{len(paths)}: {os.path.basename(p)} ({len(d)} frames)")
        return pd.concat(parts, ignore_index=True)


class LiveSniffSource:
    """Live capture from a monitor-mode interface using scapy.

    Returns the same frame table as CsvReplaySource (label is 'Unknown').
    NOT used in the demo and untested so far. Only capture on a network you
    own or are authorised to test."""

    def __init__(self, iface, seconds=5):
        self.iface = iface
        self.seconds = seconds

    def frames(self):
        from scapy.all import Dot11, RadioTap, sniff

        rows = []

        def handle(pkt):
            if not pkt.haslayer(Dot11):
                return
            d = pkt[Dot11]
            sig = getattr(pkt[RadioTap], "dBm_AntSignal", None) if pkt.haslayer(RadioTap) else None
            rows.append(dict(file="live", ts=float(pkt.time), ftype=d.type,
                             subtype=d.subtype, ta=d.addr2, ra=d.addr1,
                             signal=sig, length=len(pkt), label="Unknown"))

        sniff(iface=self.iface, prn=handle, timeout=self.seconds, store=False)
        return pd.DataFrame(rows, columns=FRAME_COLS)
