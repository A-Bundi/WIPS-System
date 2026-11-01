"""Persistent history of runs and flagged devices (SQLite, standard library only).
Stored in results/wips.db. The dashboard's Start button never deletes it."""
import os
import sqlite3
import time

import pandas as pd

DB = "results/wips.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started TEXT, ended TEXT, source TEXT, state TEXT,
    frames INTEGER DEFAULT 0, windows INTEGER DEFAULT 0,
    flagged_windows INTEGER DEFAULT 0, alerts INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS flagged_devices (
    run_id INTEGER, mac TEXT,
    first_sec INTEGER, last_sec INTEGER,
    flagged_windows INTEGER, deauth_frames INTEGER,
    peak_rate_fps REAL, signal_sum REAL, blocklisted INTEGER DEFAULT 0,
    PRIMARY KEY (run_id, mac));
"""


def connect(path=DB):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    con = sqlite3.connect(path, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")   # lets the dashboard read while detect.py writes
    con.executescript(SCHEMA)
    return con


def start_run(con, source):
    cur = con.execute("INSERT INTO runs(started, source, state) VALUES (?, ?, 'running')",
                      (time.strftime("%Y-%m-%d %H:%M:%S"), source))
    con.commit()
    return cur.lastrowid


def record_flagged(con, run_id, sec, flagged, blocked):
    """flagged: the windows the model labelled as attacks in this second."""
    if flagged.empty:
        return
    f = flagged.assign(mac=flagged["ta"].astype(str).str.lower())
    agg = f.groupby("mac").agg(fw=("mac", "size"), deauth=("deauth_count", "sum"),
                               peak=("rate_fps", "max"), sig=("mean_signal", "sum")).reset_index()
    for r in agg.itertuples(index=False):
        con.execute("""
            INSERT INTO flagged_devices(run_id, mac, first_sec, last_sec, flagged_windows,
                                        deauth_frames, peak_rate_fps, signal_sum, blocklisted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, mac) DO UPDATE SET
                last_sec = excluded.last_sec,
                flagged_windows = flagged_windows + excluded.flagged_windows,
                deauth_frames = deauth_frames + excluded.deauth_frames,
                peak_rate_fps = MAX(peak_rate_fps, excluded.peak_rate_fps),
                signal_sum = signal_sum + excluded.signal_sum,
                blocklisted = MAX(blocklisted, excluded.blocklisted)""",
                    (run_id, r.mac, int(sec), int(sec), int(r.fw), int(r.deauth),
                     float(r.peak), float(r.sig), int(r.mac in blocked)))
    con.commit()


def finish_run(con, run_id, state, frames, windows, flagged_windows, alerts):
    con.execute("""UPDATE runs SET ended=?, state=?, frames=?, windows=?,
                   flagged_windows=?, alerts=? WHERE id=?""",
                (time.strftime("%Y-%m-%d %H:%M:%S"), state, int(frames), int(windows),
                 int(flagged_windows), int(alerts), run_id))
    con.commit()


def runs_df(con):
    return pd.read_sql_query("""
        SELECT r.id AS run, r.started, r.source, r.state, r.frames, r.windows,
               r.flagged_windows,
               (SELECT COUNT(*) FROM flagged_devices d WHERE d.run_id = r.id) AS devices_flagged,
               r.alerts
        FROM runs r ORDER BY r.id DESC""", con)


def devices_all(con):
    """One row per MAC, summed over every saved run."""
    return pd.read_sql_query("""
        SELECT d.mac AS mac,
               COUNT(DISTINCT d.run_id) AS runs_flagged,
               SUM(d.flagged_windows) AS flagged_windows,
               SUM(d.deauth_frames) AS deauth_frames,
               MAX(d.peak_rate_fps) AS peak_rate_fps,
               ROUND(SUM(d.signal_sum) / SUM(d.flagged_windows), 1) AS avg_signal_dbm,
               CASE WHEN MAX(d.blocklisted) = 1 THEN 'yes' ELSE 'no' END AS blocklisted,
               MIN(r.started) AS first_seen, MAX(r.started) AS last_seen
        FROM flagged_devices d JOIN runs r ON r.id = d.run_id
        GROUP BY d.mac ORDER BY flagged_windows DESC""", con)


def devices_for_run(con, run_id):
    return pd.read_sql_query("""
        SELECT mac, first_sec, last_sec, flagged_windows, deauth_frames, peak_rate_fps,
               ROUND(signal_sum / flagged_windows, 1) AS avg_signal_dbm,
               CASE WHEN blocklisted = 1 THEN 'yes' ELSE 'no' END AS blocklisted
        FROM flagged_devices WHERE run_id = ? ORDER BY flagged_windows DESC""",
                             con, params=(int(run_id),))


def clear_history(con):
    con.execute("DELETE FROM flagged_devices")
    con.execute("DELETE FROM runs")
    con.commit()


if __name__ == "__main__":      # quick look from the terminal:  python src/store.py
    c = connect()
    pd.set_option("display.width", 200)
    print(runs_df(c).to_string(index=False), "\n")
    print(devices_all(c).to_string(index=False))
