import os
import sys

_venv_site = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "Lib", "site-packages"))
if os.path.exists(_venv_site) and _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
import joblib

def export():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "airline-passengers.csv")
    df = pd.read_csv(data_path)

    # Train / Test split (last 12 months as test set)
    train = df.iloc[:-12].copy()
    test = df.iloc[-12:].copy()

    train_log = np.log(train["Passengers"])
    test_log = np.log(test["Passengers"])

    # Fit SARIMAX(1,1,1)x(1,1,1,12) on train split for benchmark evaluation
    print("Training SARIMAX model on training set (1949-1959)...")
    sarimax_model = SARIMAX(train_log, order=[1, 1, 1], seasonal_order=(1, 1, 1, 12))
    trained_model = sarimax_model.fit(disp=False, maxiter=200)

    # Validate test set predictions
    forecast_log = trained_model.forecast(steps=12)
    forecast_sarimax = np.exp(forecast_log)

    mse = mean_squared_error(test["Passengers"], forecast_sarimax)
    mae = mean_absolute_error(test["Passengers"], forecast_sarimax)
    rmse = root_mean_squared_error(test["Passengers"], forecast_sarimax)

    print(f"\nModel Evaluation on Test Set (1960):")
    print(f"  MSE:  {mse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")

    # Also fit on full dataset (1949-1960) for production forward forecasting
    print("\nTraining SARIMAX model on full dataset for forward forecasting...")
    full_log = np.log(df["Passengers"])
    full_sarimax = SARIMAX(full_log, order=[1, 1, 1], seasonal_order=(1, 1, 1, 12))
    trained_full_model = full_sarimax.fit(disp=False, maxiter=200)

    # Save artifacts
    model_path = os.path.join(base_dir, "model.joblib")
    full_model_path = os.path.join(base_dir, "model_full.joblib")

    joblib.dump(trained_model, model_path)
    joblib.dump(trained_full_model, full_model_path)

    print(f"\nArtifacts successfully serialized:")
    print(f"  - Baseline/Test Model: {model_path}")
    print(f"  - Full Forecasting Model: {full_model_path}")

if __name__ == "__main__":
    export()
