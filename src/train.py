"""Train and evaluate the deauth-flood classifier (replay of AWID3 CSVs)."""
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

from features import FEATURES_ALL, FEATURES_NO_DEAUTH, WINDOW, build_windows
from sources import CsvReplaySource

CACHE = "data/windows.pkl"
RULE_THRESHOLD = 3  # baseline rule: >= 3 deauth frames from one device in one window
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)


def load_windows():
    if os.path.exists(CACHE) and "--rebuild" not in sys.argv:
        return pd.read_pickle(CACHE)
    print("Reading CSVs and building windows (first run only)...")
    w = build_windows(CsvReplaySource("data/raw").frames())
    w.to_pickle(CACHE)
    return w


def score(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    return dict(accuracy=float(accuracy_score(y_true, y_pred)),
                precision=float(p), recall=float(r), f1=float(f1))


def save_cm(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Deauth"]).plot(
        ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return cm.tolist()


if __name__ == "__main__":
    w = load_windows()
    # Group by 2-second blocks so neighbouring windows never land in both sets.
    w["group"] = w["file"] + "_" + (w["t0"] // 2).astype("int64").astype(str)
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    tr_idx, te_idx = next(splitter.split(w, w["label"], w["group"]))
    train, test = w.iloc[tr_idx], w.iloc[te_idx]
    y_tr, y_te = train["label"], test["label"]
    assert y_tr.sum() > 0 and y_te.sum() > 0, "attack windows missing from a split"

    print(f"\nWindows: {len(w)} total | attack windows: {int(w['label'].sum())}")
    print(f"Train: {len(train)} ({int(y_tr.sum())} attack) | "
          f"Test: {len(test)} ({int(y_te.sum())} attack)\n")

    results = []

    def record(exp, name, y_pred):
        tag = f"{exp}_{name}"
        m = score(y_te, y_pred)
        cm = save_cm(y_te, y_pred, f"{name} ({exp})", f"results/cm_{tag}.png")
        results.append(dict(experiment=exp, model=name, **m, confusion=cm))

    experiments = [("A_all_features", FEATURES_ALL),
                   ("B_no_deauth_count", FEATURES_NO_DEAUTH)]
    for exp, feats in experiments:
        if exp.startswith("A"):
            record(exp, "ThresholdRule", (test["deauth_count"] >= RULE_THRESHOLD).astype(int))
        models = {
            "DecisionTree": DecisionTreeClassifier(
                max_depth=5, class_weight="balanced", random_state=42),
            "RandomForest": RandomForestClassifier(
                n_estimators=100, max_depth=8, class_weight="balanced",
                n_jobs=-1, random_state=42),
        }
        for name, model in models.items():
            model.fit(train[feats], y_tr)
            record(exp, name, model.predict(test[feats]))
            if name == "RandomForest":
                joblib.dump({"model": model, "features": feats, "window": WINDOW},
                            f"models/rf_{exp}.joblib")
                imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
                print(f"Feature importance ({exp}):\n{imp.round(3).to_string()}\n")

    table = pd.DataFrame(results)[["experiment", "model", "accuracy", "precision", "recall", "f1"]]
    print(table.round(4).to_string(index=False))
    for r in results:
        print(f"\nConfusion [[TN FP][FN TP]] {r['experiment']} / {r['model']}: {r['confusion']}")

    with open("results/metrics.json", "w") as fh:
        json.dump(dict(window_s=WINDOW, rule_threshold=RULE_THRESHOLD,
                       n_train=len(train), n_test=len(test),
                       attack_train=int(y_tr.sum()), attack_test=int(y_te.sum()),
                       results=results), fh, indent=2)
    print("\nSaved: results/metrics.json, results/cm_*.png, models/rf_*.joblib")
