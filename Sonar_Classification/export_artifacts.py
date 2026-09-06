import os
import sys

_venv_site = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "Lib", "site-packages"))
if os.path.exists(_venv_site) and _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

def export():
    data_path = os.path.join(os.path.dirname(__file__), "sonardata.csv")
    df = pd.read_csv(data_path, header=None)
    X = df.drop(columns=[60])
    y = df[60].map({"R": 0, "M": 1})

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(probability=True, random_state=42)
    model.fit(X_scaled, y)

    out_dir = os.path.dirname(__file__)
    model_path = os.path.join(out_dir, "model.joblib")
    scaler_path = os.path.join(out_dir, "scaler.joblib")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Artifacts successfully saved to:\n  - {model_path}\n  - {scaler_path}")

if __name__ == "__main__":
    export()
