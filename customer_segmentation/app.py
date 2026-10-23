# import streamlit as st
# import pandas as pd
# import joblib

# st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="centered")

# @st.cache_resource
# def load_assets():
#     return joblib.load("model.joblib"), joblib.load("scaler.joblib")

# try:
#     model, scaler = load_assets()
# except Exception:
#     import os
#     base = os.path.dirname(__file__)
#     model = joblib.load(os.path.join(base, "model.joblib"))
#     scaler = joblib.load(os.path.join(base, "scaler.joblib"))

# st.title("🛍️ Customer Segmentation Predictor")
# st.write("Predict customer marketing persona in real time from spending habits.")

# income = st.slider("Annual Income (k$)", min_value=10, max_value=150, value=60, step=1)
# spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)

# if st.button("Segment Customer", type="primary"):
#     df = pd.DataFrame([[income, spending]], columns=["Annual Income (k$)", "Spending Score (1-100)"])
#     cluster = int(model.predict(scaler.transform(df))[0])
#     personas = {
#         0: ("Balanced Spender", "Moderate income & moderate spending habits."),
#         1: ("Conservative High-Earner", "High income but disciplined, low spending."),
#         2: ("VIP / High Roller", "High income & high spending — prime target audience!"),
#         3: ("Budget Conscious", "Low income & frugal, cautious spending."),
#         4: ("Aspirational Spender", "Low income but passionate, high spending habits.")
#     }
#     label, desc = personas.get(cluster, (f"Cluster {cluster}", "General customer segment."))
#     st.success(f"**Segment:** {label} (Cluster #{cluster})")
#     st.info(desc)
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")

@st.cache_resource
def load_assets():
    base = os.path.dirname(__file__)
    # Fallback to local directory if __file__ pathing fails during deployment
    model_path = os.path.join(base, "model.joblib") if os.path.exists(os.path.join(base, "model.joblib")) else "model.joblib"
    scaler_path = os.path.join(base, "scaler.joblib") if os.path.exists(os.path.join(base, "scaler.joblib")) else "scaler.joblib"
    
    return joblib.load(model_path), joblib.load(scaler_path)

model, scaler = load_assets()

personas = {
    0: ("Balanced Spender", "Moderate income & moderate spending habits."),
    1: ("Conservative High-Earner", "High income but disciplined, low spending."),
    2: ("VIP / High Roller", "High income & high spending — prime target audience!"),
    3: ("Budget Conscious", "Low income & frugal, cautious spending."),
    4: ("Aspirational Spender", "Low income but passionate, high spending habits.")
}

st.title("🛍️ Advanced Customer Segmentation Predictor")
st.markdown("Predict customer marketing personas in real-time or via batch CSV uploads.")

# Create a tabbed interface
tab1, tab2, tab3 = st.tabs(["🎯 Real-Time Prediction", "📁 Batch Upload", "📊 Persona Insights"])

with tab1:
    st.subheader("Individual Customer Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        income = st.slider("Annual Income (k$)", min_value=10, max_value=150, value=60, step=1)
    with col2:
        spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)

    if st.button("Segment Customer", type="primary"):
        df = pd.DataFrame([[income, spending]], columns=["Annual Income (k$)", "Spending Score (1-100)"])
        cluster = int(model.predict(scaler.transform(df))[0])
        label, desc = personas.get(cluster, (f"Cluster {cluster}", "General customer segment."))
        
        st.success(f"**Segment:** {label} (Cluster #{cluster})")
        st.info(desc)

with tab2:
    st.subheader("Batch Customer Segmentation")
    st.write("Upload a CSV file containing customer data to segment an entire database at once.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        # Check if required columns exist
        if "Annual Income (k$)" in batch_df.columns and "Spending Score (1-100)" in batch_df.columns:
            X_batch = batch_df[["Annual Income (k$)", "Spending Score (1-100)"]]
            
            # Predict and map labels
            predictions = model.predict(scaler.transform(X_batch))
            batch_df["Cluster"] = predictions
            batch_df["Persona"] = batch_df["Cluster"].map(lambda x: personas.get(x, ("Unknown",))[0])
            
            st.dataframe(batch_df, use_container_width=True)
            
            # Allow user to download the new dataset
            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Segmented Data",
                data=csv,
                file_name="segmented_customers.csv",
                mime="text/csv"
            )
        else:
            st.error("Error: CSV must contain 'Annual Income (k$)' and 'Spending Score (1-100)' columns.")

with tab3:
    st.subheader("Cluster Profiles")
    st.write("Understanding the 5 distinct customer cohorts discovered through our geometric K-Means partitioning[cite: 2]:")
    
    for cluster_id, (label, desc) in personas.items():
        st.markdown(f"**Cluster {cluster_id}: {label}**")
        st.write(f"- *Insight:* {desc}[cite: 2]")
