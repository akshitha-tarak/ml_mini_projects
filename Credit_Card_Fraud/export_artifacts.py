import os
import sys

_venv_site = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "Lib", "site-packages"))
if os.path.exists(_venv_site) and _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score
import joblib

def export():
    csv_path = os.path.join(os.path.dirname(__file__), "creditcard.csv")
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Features: Time, V1-V28 (Amount and Class dropped as in notebook)
    X = df.drop(columns=["Class", "Amount"])
    y = df["Class"]

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Balanced Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        max_features="log2",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_test_prob)
    print(f"Test PR-AUC: {pr_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred, digits=4))

    out_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    joblib.dump(model, out_path, compress=3)
    print(f"Model successfully saved to: {out_path} (Size: {os.path.getsize(out_path)/(1024*1024):.2f} MB)")

if __name__ == "__main__":
    export()
