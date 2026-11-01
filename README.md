# WIPS: ML-based Wi-Fi deauthentication-flood detection and prevention

Wireless intrusion prevention prototype. It reads 802.11 frames (replayed AWID3 CSVs; a live
monitor-mode source is included but untested), summarises each device in 0.25 s windows,
classifies each window with a scikit-learn Random Forest, and on an attack raises an alert,
logs it and adds the offending MAC to a blocklist. A Streamlit dashboard shows it live.

## Setup (Kali / Debian)
    sudo apt install -y python3-full python3-venv python3-pip git
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Data (not included)
    # needs a Kaggle API token in ~/.kaggle/
    kaggle datasets download -d anikatab/awid3-5csv -p data/raw --unzip

## Reproduce
    python src/train.py          # builds windows cache, single-split evaluation
    python src/crossval.py       # cross-validated results used on the poster
    python src/train_final.py    # trains the deployment model (models/wips_rf.joblib)

## Run
    streamlit run src/dashboard.py                                   # dashboard (replay)
    python src/detect.py --csv data/raw/deauth/Deauth_25.csv --delay 0.3   # terminal only

Alerts go to results/alerts.log, blocked MACs to results/blocklist.txt, and runs plus flagged
devices are saved in results/wips.db (view with `python src/store.py`).

## Layout
- `src/sources.py` CSV replay and live-capture sources (same frame table)
- `src/features.py` windowing and the 8 features
- `src/train.py`, `crossval.py`, `train_final.py` training and evaluation
- `src/detect.py`, `prevent.py`, `store.py` detection, response, history
- `src/dashboard.py` Streamlit dashboard
- `results/` metrics, confusion matrices and inspection outputs

## Results (grouped 4-fold CV, 5 seeds, attack = positive)
| Model | Precision | Recall | F1 |
|---|---|---|---|
| A, threshold rule | 1.00 | 1.00 | 1.00 |
| A, Decision Tree | 1.00 | 0.97 | 0.98 |
| A, Random Forest | 1.00 | 0.94 | 0.97 |
| B (deauth counts hidden), Decision Tree | 0.32 | 0.90 | 0.48 |
| B (deauth counts hidden), Random Forest | 0.82 | 0.83 | 0.82 |

Limitations: one 14 s attack (32 attack windows); the normal data has no legitimate deauth
bursts; the demo replays data the final model trained on; blocklisted MACs are the claimed
(possibly spoofed) source; live capture is untested.

## Ethics
Only recorded data is replayed. Any live capture or attack generation must stay on your own
or an approved testbed.

## Use of AI assistants
Built with help from Claude (Anthropic) for design, code and debugging; all code was run and
checked by the author.
