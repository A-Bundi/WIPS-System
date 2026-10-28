"""Grouped cross-validation: every window is tested once; repeated over 5 seeds."""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

from features import FEATURES_ALL, FEATURES_NO_DEAUTH
from train import RULE_THRESHOLD, load_windows, save_cm, score

SEEDS = [0, 1, 2, 3, 4]
EXPS = [("A_all_features", FEATURES_ALL), ("B_no_deauth_count", FEATURES_NO_DEAUTH)]


def make(name):
    if name == "DecisionTree":
        return DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
    return RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced",
                                  n_jobs=-1, random_state=42)


w = load_windows()
y = w["label"].values
groups = (w["file"] + "_" + (w["t0"] // 2).astype("int64").astype(str)).values
print(f"{len(w)} windows, {int(y.sum())} attack windows\n")

runs, cms = {}, {}

rule = (w["deauth_count"] >= RULE_THRESHOLD).astype(int).values
runs[("A_all_features", "ThresholdRule")] = [score(y, rule)] * len(SEEDS)
cms[("A_all_features", "ThresholdRule")] = save_cm(
    y, rule, "ThresholdRule (A, CV)", "results/cv_cm_A_all_features_ThresholdRule.png")

for seed in SEEDS:
    splits = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
                  .split(w, y, groups))
    if seed == SEEDS[0]:
        print("attack windows per test fold (seed 0):", [int(y[te].sum()) for _, te in splits], "\n")
    for exp, feats in EXPS:
        X = w[feats].values
        for name in ("DecisionTree", "RandomForest"):
            pred = np.zeros(len(w), dtype=int)
            for tr, te in splits:
                pred[te] = make(name).fit(X[tr], y[tr]).predict(X[te])
            runs.setdefault((exp, name), []).append(score(y, pred))
            if seed == SEEDS[0]:
                cms[(exp, name)] = save_cm(y, pred, f"{name} ({exp[0]}, CV)",
                                           f"results/cv_cm_{exp}_{name}.png")

rows = []
for (exp, name), ms in runs.items():
    d = pd.DataFrame(ms)
    rows.append(dict(experiment=exp, model=name,
                     **{f"{c}_mean": d[c].mean() for c in d},
                     **{f"{c}_std": d[c].std(ddof=0) for c in d},
                     confusion_seed0=cms[(exp, name)]))
out = pd.DataFrame(rows)
show = out[["experiment", "model", "accuracy_mean", "precision_mean", "precision_std",
            "recall_mean", "recall_std", "f1_mean", "f1_std"]]
print(show.round(4).to_string(index=False))
for r in rows:
    print(f"\nConfusion [[TN FP][FN TP]] (seed 0) {r['experiment']} / {r['model']}: {r['confusion_seed0']}")

with open("results/cv_metrics.json", "w") as fh:
    json.dump(rows, fh, indent=2)
print("\nSaved: results/cv_metrics.json, results/cv_cm_*.png")
