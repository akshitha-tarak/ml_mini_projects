import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="centered")

@st.cache_resource
def load_assets():
    return joblib.load("model.joblib"), joblib.load("scaler.joblib")

try:
    model, scaler = load_assets()
except Exception:
    import os
    base = os.path.dirname(__file__)
    model = joblib.load(os.path.join(base, "model.joblib"))
    scaler = joblib.load(os.path.join(base, "scaler.joblib"))

st.title("🛍️ Customer Segmentation Predictor")
st.write("Predict customer marketing persona in real time from spending habits.")

income = st.slider("Annual Income (k$)", min_value=10, max_value=150, value=60, step=1)
spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)

if st.button("Segment Customer", type="primary"):
    df = pd.DataFrame([[income, spending]], columns=["Annual Income (k$)", "Spending Score (1-100)"])
    cluster = int(model.predict(scaler.transform(df))[0])
    personas = {
        0: ("Balanced Spender", "Moderate income & moderate spending habits."),
        1: ("Conservative High-Earner", "High income but disciplined, low spending."),
        2: ("VIP / High Roller", "High income & high spending — prime target audience!"),
        3: ("Budget Conscious", "Low income & frugal, cautious spending."),
        4: ("Aspirational Spender", "Low income but passionate, high spending habits.")
    }
    label, desc = personas.get(cluster, (f"Cluster {cluster}", "General customer segment."))
    st.success(f"**Segment:** {label} (Cluster #{cluster})")
    st.info(desc)
