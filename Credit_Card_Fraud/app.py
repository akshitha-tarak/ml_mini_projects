import streamlit as st
import pandas as pd
import joblib, os

st.set_page_config(page_title="Fraud Detection Engine", page_icon="💳", layout="centered")

@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    return joblib.load(os.path.join(base, "model.joblib"))

model = load_model()

st.title("💳 Credit Card Fraud Detection")
st.write("Real-time transaction risk scoring using a Balanced Random Forest model.")

cols = ["Time"] + [f"V{i}" for i in range(1, 29)]
preset = st.radio("Select Scenario:", ["Normal Purchase", "High-Risk Fraud Pattern"], horizontal=True)

values = [0.0] * 29
if preset == "High-Risk Fraud Pattern":
    values[14], values[17], values[4] = -4.5, -5.2, 3.8

values[14] = st.slider("Feature V14 (Fraud Indicator)", -10.0, 5.0, float(values[14]), 0.1)
values[17] = st.slider("Feature V17 (Fraud Indicator)", -10.0, 5.0, float(values[17]), 0.1)

if st.button("Evaluate Transaction", type="primary"):
    tx_df = pd.DataFrame([values], columns=cols)
    is_fraud = int(model.predict(tx_df)[0])
    prob = model.predict_proba(tx_df)[0][1] if hasattr(model, "predict_proba") else (1.0 if is_fraud else 0.0)
    if is_fraud == 1:
        st.error(f"🚨 **FRAUD ALERT! Transaction Blocked** (Risk Score: {prob:.1%})")
    else:
        st.success(f"✅ **Transaction Approved (Legitimate)** (Risk Score: {prob:.1%})")
