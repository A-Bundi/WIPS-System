"""Train the deployment model on all windows (evaluation numbers come from crossval.py)."""
import joblib
from sklearn.ensemble import RandomForestClassifier

from features import FEATURES_ALL, WINDOW
from train import load_windows

w = load_windows()
model = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced",
                               n_jobs=-1, random_state=42).fit(w[FEATURES_ALL], w["label"])
joblib.dump({"model": model, "features": FEATURES_ALL, "window": WINDOW}, "models/wips_rf.joblib")
print("Saved models/wips_rf.joblib")
