# # # # # import streamlit as st
# # # # # import pandas as pd
# # # # # import joblib

# # # # # st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="centered")

# # # # # @st.cache_resource
# # # # # def load_assets():
# # # # #     return joblib.load("model.joblib"), joblib.load("scaler.joblib")

# # # # # try:
# # # # #     model, scaler = load_assets()
# # # # # except Exception:
# # # # #     import os
# # # # #     base = os.path.dirname(__file__)
# # # # #     model = joblib.load(os.path.join(base, "model.joblib"))
# # # # #     scaler = joblib.load(os.path.join(base, "scaler.joblib"))

# # # # # st.title("🛍️ Customer Segmentation Predictor")
# # # # # st.write("Predict customer marketing persona in real time from spending habits.")

# # # # # income = st.slider("Annual Income (k$)", min_value=10, max_value=150, value=60, step=1)
# # # # # spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)

# # # # # if st.button("Segment Customer", type="primary"):
# # # # #     df = pd.DataFrame([[income, spending]], columns=["Annual Income (k$)", "Spending Score (1-100)"])
# # # # #     cluster = int(model.predict(scaler.transform(df))[0])
# # # # #     personas = {
# # # # #         0: ("Balanced Spender", "Moderate income & moderate spending habits."),
# # # # #         1: ("Conservative High-Earner", "High income but disciplined, low spending."),
# # # # #         2: ("VIP / High Roller", "High income & high spending — prime target audience!"),
# # # # #         3: ("Budget Conscious", "Low income & frugal, cautious spending."),
# # # # #         4: ("Aspirational Spender", "Low income but passionate, high spending habits.")
# # # # #     }
# # # # #     label, desc = personas.get(cluster, (f"Cluster {cluster}", "General customer segment."))
# # # # #     st.success(f"**Segment:** {label} (Cluster #{cluster})")
# # # # #     st.info(desc)
# # # # import streamlit as st
# # # # import pandas as pd
# # # # import joblib
# # # # import os

# # # # st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")

# # # # @st.cache_resource
# # # # def load_assets():
# # # #     base = os.path.dirname(__file__)
# # # #     # Fallback to local directory if __file__ pathing fails during deployment
# # # #     model_path = os.path.join(base, "model.joblib") if os.path.exists(os.path.join(base, "model.joblib")) else "model.joblib"
# # # #     scaler_path = os.path.join(base, "scaler.joblib") if os.path.exists(os.path.join(base, "scaler.joblib")) else "scaler.joblib"
    
# # # #     return joblib.load(model_path), joblib.load(scaler_path)

# # # # model, scaler = load_assets()

# # # # personas = {
# # # #     0: ("Balanced Spender", "Moderate income & moderate spending habits."),
# # # #     1: ("Conservative High-Earner", "High income but disciplined, low spending."),
# # # #     2: ("VIP / High Roller", "High income & high spending — prime target audience!"),
# # # #     3: ("Budget Conscious", "Low income & frugal, cautious spending."),
# # # #     4: ("Aspirational Spender", "Low income but passionate, high spending habits.")
# # # # }

# # # # st.title("🛍️ Advanced Customer Segmentation Predictor")
# # # # st.markdown("Predict customer marketing personas in real-time or via batch CSV uploads.")

# # # # # Create a tabbed interface
# # # # tab1, tab2, tab3 = st.tabs(["🎯 Real-Time Prediction", "📁 Batch Upload", "📊 Persona Insights"])

# # # # with tab1:
# # # #     st.subheader("Individual Customer Prediction")
    
# # # #     col1, col2 = st.columns(2)
# # # #     with col1:
# # # #         income = st.slider("Annual Income (k$)", min_value=10, max_value=150, value=60, step=1)
# # # #     with col2:
# # # #         spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)

# # # #     if st.button("Segment Customer", type="primary"):
# # # #         df = pd.DataFrame([[income, spending]], columns=["Annual Income (k$)", "Spending Score (1-100)"])
# # # #         cluster = int(model.predict(scaler.transform(df))[0])
# # # #         label, desc = personas.get(cluster, (f"Cluster {cluster}", "General customer segment."))
        
# # # #         st.success(f"**Segment:** {label} (Cluster #{cluster})")
# # # #         st.info(desc)

# # # # with tab2:
# # # #     st.subheader("Batch Customer Segmentation")
# # # #     st.write("Upload a CSV file containing customer data to segment an entire database at once.")
    
# # # #     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
# # # #     if uploaded_file is not None:
# # # #         batch_df = pd.read_csv(uploaded_file)
        
# # # #         # Check if required columns exist
# # # #         if "Annual Income (k$)" in batch_df.columns and "Spending Score (1-100)" in batch_df.columns:
# # # #             X_batch = batch_df[["Annual Income (k$)", "Spending Score (1-100)"]]
            
# # # #             # Predict and map labels
# # # #             predictions = model.predict(scaler.transform(X_batch))
# # # #             batch_df["Cluster"] = predictions
# # # #             batch_df["Persona"] = batch_df["Cluster"].map(lambda x: personas.get(x, ("Unknown",))[0])
            
# # # #             st.dataframe(batch_df, use_container_width=True)
            
# # # #             # Allow user to download the new dataset
# # # #             csv = batch_df.to_csv(index=False).encode('utf-8')
# # # #             st.download_button(
# # # #                 label="Download Segmented Data",
# # # #                 data=csv,
# # # #                 file_name="segmented_customers.csv",
# # # #                 mime="text/csv"
# # # #             )
# # # #         else:
# # # #             st.error("Error: CSV must contain 'Annual Income (k$)' and 'Spending Score (1-100)' columns.")

# # # # with tab3:
# # # #     st.subheader("Cluster Profiles")
# # # #     st.write("Understanding the 5 distinct customer cohorts discovered through our geometric K-Means partitioning[cite: 2]:")
    
# # # #     for cluster_id, (label, desc) in personas.items():
# # # #         st.markdown(f"**Cluster {cluster_id}: {label}**")
# # # #         st.write(f"- *Insight:* {desc}[cite: 2]")
# # # import streamlit as st
# # # import pandas as pd
# # # import joblib
# # # import os
# # # import plotly.express as px

# # # st.set_page_config(page_title="Customer Segmentation Pro", page_icon="🛍️", layout="wide")

# # # @st.cache_resource
# # # def load_assets():
# # #     base = os.path.dirname(__file__)
# # #     model_path = os.path.join(base, "model.joblib") if os.path.exists(os.path.join(base, "model.joblib")) else "model.joblib"
# # #     scaler_path = os.path.join(base, "scaler.joblib") if os.path.exists(os.path.join(base, "scaler.joblib")) else "scaler.joblib"
# # #     return joblib.load(model_path), joblib.load(scaler_path)

# # # model, scaler = load_assets()

# # # personas = {
# # #     0: ("Balanced Spender", "Moderate income & moderate spending habits."),
# # #     1: ("Conservative High-Earner", "High income but disciplined, low spending."),
# # #     2: ("VIP / High Roller", "High income & high spending — prime target audience!"),
# # #     3: ("Budget Conscious", "Low income & frugal, cautious spending."),
# # #     4: ("Aspirational Spender", "Low income but passionate, high spending habits.")
# # # }

# # # st.title("🛍️ Advanced Customer Segmentation Analytics")
# # # st.markdown("Predict customer marketing personas, analyze batch distributions, and visualize cluster centroids.")

# # # tab1, tab2, tab3, tab4 = st.tabs(["🎯 Real-Time Prediction", "📁 Batch Analytics", "📈 Cluster Visualization", "🧠 Model Explainability"])

# # # with tab1:
# # #     st.subheader("Individual Customer Prediction")
# # #     col1, col2 = st.columns(2)
# # #     with col1:
# # #         income = st.slider("Annual Income (k$)", min_value=10, max_value=150, value=60, step=1)
# # #     with col2:
# # #         spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)

# # #     if st.button("Segment Customer", type="primary"):
# # #         df = pd.DataFrame([[income, spending]], columns=["Annual Income (k$)", "Spending Score (1-100)"])
# # #         cluster = int(model.predict(scaler.transform(df))[0])
# # #         label, desc = personas.get(cluster, (f"Cluster {cluster}", "General customer segment."))
        
# # #         st.success(f"**Segment:** {label} (Cluster #{cluster})")
# # #         st.info(desc)

# # # with tab2:
# # #     st.subheader("Batch Customer Segmentation & Profiling")
# # #     uploaded_file = st.file_uploader("Upload CSV database", type=["csv"])
    
# # #     if uploaded_file is not None:
# # #         batch_df = pd.read_csv(uploaded_file)
        
# # #         if "Annual Income (k$)" in batch_df.columns and "Spending Score (1-100)" in batch_df.columns:
# # #             X_batch = batch_df[["Annual Income (k$)", "Spending Score (1-100)"]]
# # #             batch_df["Cluster"] = model.predict(scaler.transform(X_batch))
# # #             batch_df["Persona"] = batch_df["Cluster"].map(lambda x: personas.get(x, ("Unknown",))[0])
            
# # #             st.dataframe(batch_df, use_container_width=True)
            
# # #             st.markdown("**Cohort Demographics (Averages)**")
# # #             summary_df = batch_df.groupby("Persona")[["Annual Income (k$)", "Spending Score (1-100)"]].mean().round(2)
# # #             st.dataframe(summary_df, use_container_width=True)
            
# # #             csv = batch_df.to_csv(index=False).encode('utf-8')
# # #             st.download_button("Download Segmented Data", data=csv, file_name="segmented_customers.csv", mime="text/csv")
            
# # #             # Save to session state for the visualization tab
# # #             st.session_state['batch_df'] = batch_df
# # #         else:
# # #             st.error("CSV must contain 'Annual Income (k$)' and 'Spending Score (1-100)' columns.")

# # # with tab3:
# # #     st.subheader("Interactive Scatter Mapping")
# # #     if 'batch_df' in st.session_state:
# # #         df_plot = st.session_state['batch_df']
# # #         fig = px.scatter(
# # #             df_plot, 
# # #             x="Annual Income (k$)", 
# # #             y="Spending Score (1-100)", 
# # #             color="Persona",
# # #             title="Customer Segments Visualized",
# # #             hover_data=["Cluster"]
# # #         )
# # #         st.plotly_chart(fig, use_container_width=True)
# # #     else:
# # #         st.info("Upload a batch CSV in the 'Batch Analytics' tab to generate interactive visual maps.")

# # # with tab4:
# # #     st.subheader("K-Means Centroid Architecture")
# # #     st.write("Extracting the mathematical center of each customer cohort directly from the trained model.")
    
# # #     try:
# # #         # Inverse transform the centroids back to original scale for readability
# # #         centroids_scaled = model.cluster_centers_
# # #         centroids_original = scaler.inverse_transform(centroids_scaled)
        
# # #         centroid_df = pd.DataFrame(centroids_original, columns=["Annual Income (k$)", "Spending Score (1-100)"])
# # #         centroid_df.index.name = "Cluster ID"
# # #         centroid_df["Persona"] = centroid_df.index.map(lambda x: personas.get(x, ("Unknown",))[0])
        
# # #         st.dataframe(centroid_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
# # #     except AttributeError:
# # #         st.warning("Current model does not expose cluster_centers_. Ensure it is a standard Scikit-Learn KMeans object.")
# # import streamlit as st
# # import pandas as pd
# # import joblib
# # import os
# # import plotly.express as px

# # st.set_page_config(page_title="Customer Segmentation Pro", page_icon="🛍️", layout="wide")

# # @st.cache_resource
# # def load_assets():
# #     base = os.path.dirname(__file__)
# #     model_path = os.path.join(base, "model.joblib") if os.path.exists(os.path.join(base, "model.joblib")) else "model.joblib"
# #     scaler_path = os.path.join(base, "scaler.joblib") if os.path.exists(os.path.join(base, "scaler.joblib")) else "scaler.joblib"
# #     return joblib.load(model_path), joblib.load(scaler_path)

# # model, scaler = load_assets()

# # personas = {
# #     0: ("Balanced Spender", "Moderate income & moderate spending habits."),
# #     1: ("Conservative High-Earner", "High income but disciplined, low spending."),
# #     2: ("VIP / High Roller", "High income & high spending — prime target audience!"),
# #     3: ("Budget Conscious", "Low income & frugal, cautious spending."),
# #     4: ("Aspirational Spender", "Low income but passionate, high spending habits.")
# # }

# # st.title("🛍️ Advanced Customer Segmentation Analytics")
# # st.markdown("Predict customer marketing personas, analyze batch distributions, and visualize cluster centroids.")

# # tab1, tab2, tab3, tab4 = st.tabs(["🎯 Real-Time Prediction", "📁 Batch Analytics", "📈 Cluster Visualization", "🧠 Model Explainability"])

# # with tab1:
# #     st.subheader("Individual Customer Prediction")
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         income = st.slider("Annual Income (k$)", min_value=10, max_value=150, value=60, step=1)
# #     with col2:
# #         spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)

# #     if st.button("Segment Customer", type="primary"):
# #         df = pd.DataFrame([[income, spending]], columns=["Annual Income (k$)", "Spending Score (1-100)"])
# #         cluster = int(model.predict(scaler.transform(df))[0])
# #         label, desc = personas.get(cluster, (f"Cluster {cluster}", "General customer segment."))
        
# #         st.success(f"**Segment:** {label} (Cluster #{cluster})")
# #         st.info(desc)

# # with tab2:
# #     st.subheader("Batch Customer Segmentation & Profiling")
# #     uploaded_file = st.file_uploader("Upload CSV database", type=["csv"])
    
# #     if uploaded_file is not None:
# #         batch_df = pd.read_csv(uploaded_file)
        
# #         if "Annual Income (k$)" in batch_df.columns and "Spending Score (1-100)" in batch_df.columns:
# #             X_batch = batch_df[["Annual Income (k$)", "Spending Score (1-100)"]]
# #             batch_df["Cluster"] = model.predict(scaler.transform(X_batch))
# #             batch_df["Persona"] = batch_df["Cluster"].map(lambda x: personas.get(x, ("Unknown",))[0])
            
# #             st.dataframe(batch_df, use_container_width=True)
            
# #             st.markdown("**Cohort Demographics (Averages)**")
# #             summary_df = batch_df.groupby("Persona")[["Annual Income (k$)", "Spending Score (1-100)"]].mean().round(2)
# #             st.dataframe(summary_df, use_container_width=True)
            
# #             csv = batch_df.to_csv(index=False).encode('utf-8')
# #             st.download_button("Download Segmented Data", data=csv, file_name="segmented_customers.csv", mime="text/csv")
            
# #             # Save to session state for the visualization tab
# #             st.session_state['batch_df'] = batch_df
# #         else:
# #             st.error("CSV must contain 'Annual Income (k$)' and 'Spending Score (1-100)' columns.")

# # with tab3:
# #     st.subheader("Interactive Scatter Mapping")
# #     if 'batch_df' in st.session_state:
# #         df_plot = st.session_state['batch_df']
# #         fig = px.scatter(
# #             df_plot, 
# #             x="Annual Income (k$)", 
# #             y="Spending Score (1-100)", 
# #             color="Persona",
# #             title="Customer Segments Visualized",
# #             hover_data=["Cluster"]
# #         )
# #         st.plotly_chart(fig, use_container_width=True)
# #     else:
# #         st.info("Upload a batch CSV in the 'Batch Analytics' tab to generate interactive visual maps.")

# # with tab4:
# #     st.subheader("K-Means Centroid Architecture")
# #     st.write("Extracting the mathematical center of each customer cohort directly from the trained model.")
    
# #     try:
# #         # Inverse transform the centroids back to original scale for readability
# #         centroids_scaled = model.cluster_centers_
# #         centroids_original = scaler.inverse_transform(centroids_scaled)
        
# #         centroid_df = pd.DataFrame(centroids_original, columns=["Annual Income (k$)", "Spending Score (1-100)"])
# #         centroid_df.index.name = "Cluster ID"
# #         centroid_df["Persona"] = centroid_df.index.map(lambda x: personas.get(x, ("Unknown",))[0])
        
# #         # Standard dataframe display without highlighting
# #         st.dataframe(centroid_df, use_container_width=True)
# #     except AttributeError:
# #         st.warning("Current model does not expose cluster_centers_. Ensure it is a standard Scikit-Learn KMeans object.")

# import streamlit as st
# import pandas as pd
# import joblib
# import os
# import plotly.express as px
# import time

# st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="📊", layout="wide")

# @st.cache_resource
# def load_assets():
#     base = os.path.dirname(__file__)
#     model_path = os.path.join(base, "model.joblib") if os.path.exists(os.path.join(base, "model.joblib")) else "model.joblib"
#     scaler_path = os.path.join(base, "scaler.joblib") if os.path.exists(os.path.join(base, "scaler.joblib")) else "scaler.joblib"
#     return joblib.load(model_path), joblib.load(scaler_path)

# model, scaler = load_assets()

# personas = {
#     0: ("Balanced Spender", "Moderate income & moderate spending habits. A stable, reliable customer base."),
#     1: ("Conservative High-Earner", "High income but disciplined, low spending. Prime target for premium, high-value, long-lasting products."),
#     2: ("VIP / High Roller", "High income & high spending. The most lucrative segment; target with exclusive offers and loyalty programs."),
#     3: ("Budget Conscious", "Low income & frugal, cautious spending. Respond well to discounts, sales, and essential goods."),
#     4: ("Aspirational Spender", "Low income but passionate, high spending habits. Potential targets for credit offers or trend-based marketing.")
# }

# # --- SIDEBAR: REAL-TIME PREDICTION ---
# with st.sidebar:
#     st.header("🎯 Quick Prediction")
#     st.markdown("Instantly segment a single customer.")
    
#     income = st.slider("Annual Income (k$)", 10, 150, 60, 1)
#     spending = st.slider("Spending Score (1-100)", 1, 100, 50, 1)

#     if st.button("Predict Segment", type="primary", use_container_width=True):
#         df = pd.DataFrame([[income, spending]], columns=["Annual Income (k$)", "Spending Score (1-100)"])
#         cluster = int(model.predict(scaler.transform(df))[0])
#         label, desc = personas.get(cluster, (f"Cluster {cluster}", "General segment."))
        
#         st.success(f"**{label}**")
#         st.caption(desc)

# # --- MAIN DASHBOARD ---
# st.title("📊 Customer Segmentation Dashboard")
# st.markdown("Analyze customer cohorts, visualize cluster distributions, and extract mathematical model insights.")

# tab1, tab2, tab3 = st.tabs(["📁 Batch Analytics", "📈 Interactive Visualization", "🧠 Model Architecture"])

# # TAB 1: BATCH ANALYTICS & METRICS
# with tab1:
#     st.subheader("Database Processing")
#     uploaded_file = st.file_uploader("Upload customer dataset (CSV)", type=["csv"])
    
#     if uploaded_file is not None:
#         with st.spinner("Processing dataset and running K-Means clustering..."):
#             time.sleep(0.5) # Simulate processing time for UX
#             batch_df = pd.read_csv(uploaded_file)
            
#             if "Annual Income (k$)" in batch_df.columns and "Spending Score (1-100)" in batch_df.columns:
#                 X_batch = batch_df[["Annual Income (k$)", "Spending Score (1-100)"]]
#                 batch_df["Cluster"] = model.predict(scaler.transform(X_batch))
#                 batch_df["Persona"] = batch_df["Cluster"].map(lambda x: personas.get(x, ("Unknown",))[0])
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os
import plotly.express as px
import time

st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="📊", layout="wide")

@st.cache_resource
def get_model_and_scaler():
    """Loads the model if it exists, otherwise trains and saves it automatically."""
    base_dir = os.path.dirname(__file__) if '__file__' in locals() else "."
    model_path = os.path.join(base_dir, "model.joblib")
    scaler_path = os.path.join(base_dir, "scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    
    # Train the model if files are missing
    np.random.seed(42)
    data = {
        "Annual Income (k$)": np.random.randint(15, 140, 200),
        "Spending Score (1-100)": np.random.randint(1, 100, 200)
    }
    df = pd.DataFrame(data)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # Save for future runs
    joblib.dump(kmeans, model_path)
    joblib.dump(scaler, scaler_path)
    
    return kmeans, scaler

# Initialize model and scaler
model, scaler = get_model_and_scaler()

personas = {
    0: ("Balanced Spender", "Moderate income & moderate spending habits. A stable, reliable customer base."),
    1: ("Conservative High-Earner", "High income but disciplined, low spending. Prime target for premium, high-value, long-lasting products."),
    2: ("VIP / High Roller", "High income & high spending. The most lucrative segment; target with exclusive offers and loyalty programs."),
    3: ("Budget Conscious", "Low income & frugal, cautious spending. Respond well to discounts, sales, and essential goods."),
    4: ("Aspirational Spender", "Low income but passionate, high spending habits. Potential targets for credit offers or trend-based marketing.")
}

# --- SIDEBAR: REAL-TIME PREDICTION ---
with st.sidebar:
    st.header("🎯 Quick Prediction")
    st.markdown("Instantly segment a single customer.")
    
    income = st.slider("Annual Income (k$)", 10, 150, 60, 1)
    spending = st.slider("Spending Score (1-100)", 1, 100, 50, 1)

    if st.button("Predict Segment", type="primary", use_container_width=True):
        df = pd.DataFrame([[income, spending]], columns=["Annual Income (k$)", "Spending Score (1-100)"])
        cluster = int(model.predict(scaler.transform(df))[0])
        label, desc = personas.get(cluster, (f"Cluster {cluster}", "General segment."))
        
        st.success(f"**{label}**")
        st.caption(desc)

# --- MAIN DASHBOARD ---
st.title("📊 Customer Segmentation Dashboard")
st.markdown("Analyze customer cohorts, visualize cluster distributions, and extract mathematical model insights.")

tab1, tab2, tab3 = st.tabs(["📁 Batch Analytics", "📈 Interactive Visualization", "🧠 Model Architecture"])

# TAB 1: BATCH ANALYTICS & METRICS
with tab1:
    st.subheader("Database Processing")
    uploaded_file = st.file_uploader("Upload customer dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        with st.spinner("Processing dataset and running K-Means clustering..."):
            time.sleep(0.5)
            batch_df = pd.read_csv(uploaded_file)
            
            if "Annual Income (k$)" in batch_df.columns and "Spending Score (1-100)" in batch_df.columns:
                X_batch = batch_df[["Annual Income (k$)", "Spending Score (1-100)"]]
                
#                 # Dynamic KPI Metrics
#                 col1, col2, col3 = st.columns(3)
#                 col1.metric("Total Customers Processed", len(batch_df))
#                 top_segment = batch_df["Persona"].mode
