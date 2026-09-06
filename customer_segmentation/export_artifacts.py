import os
import sys

# Ensure local .venv site-packages is discoverable if running from global python
_venv_site = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "Lib", "site-packages"))
if os.path.exists(_venv_site) and _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

def export():
    data_path = os.path.join(os.path.dirname(__file__), "Mall_Customers.csv")
    df = pd.read_csv(data_path)
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=5, random_state=42, n_init=10)
    model.fit(X_scaled)

    out_dir = os.path.dirname(__file__)
    model_path = os.path.join(out_dir, "model.joblib")
    scaler_path = os.path.join(out_dir, "scaler.joblib")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Artifacts successfully saved to:\n  - {model_path}\n  - {scaler_path}")

if __name__ == "__main__":
    export()
