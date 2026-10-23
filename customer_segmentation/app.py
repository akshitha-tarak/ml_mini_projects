# # # # # # # import streamlit as st
# # # # # # # import pandas as pd
# # # # # # # import joblib

# # # # # # # st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="centered")

# # # # # # # @st.cache_resource
# # # # # # # def load_assets():
# # # # # # #     return joblib.load("model.joblib"), joblib.load("scaler.joblib")

# # # # # # # try:
# # # # # # #     model, scaler = load_assets()
# # # # # # # except Exception:
# # # # # # #     import os
# # # # # # #     base = os.path.dirname(__file__)
# # # # # # #     model = joblib.load(os.path.join(base, "model.joblib"))
# # # # # # #     scaler = joblib.load(os.path.join(base, "scaler.joblib"))

# # # # # # # st.title("🛍️ Customer Segmentation Predictor")
# # # # # # # st.write("Predict customer marketing persona in real time from spending habits.")

# # # # # # # income = st.slider("Annual Income (k$)", min_value=10, max_value=150, value=60, step=1)
# # # # # # # spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)

# # # # # # # if st.button("Segment Customer", type="primary"):
# # # # # # #     df = pd.DataFrame([[income, spending]], columns=["Annual Income (k$)", "Spending Score (1-100)"])
# # # # # # #     cluster = int(model.predict(scaler.transform(df))[0])
# # # # # # #     personas = {
# # # # # # #         0: ("Balanced Spender", "Moderate income & moderate spending habits."),
# # # # # # #         1: ("Conservative High-Earner", "High income but disciplined, low spending."),
# # # # # # #         2: ("VIP / High Roller", "High income & high spending — prime target audience!"),
# # # # # # #         3: ("Budget Conscious", "Low income & frugal, cautious spending."),
# # # # # # #         4: ("Aspirational Spender", "Low income but passionate, high spending habits.")
# # # # # # #     }
# # # # # # #     label, desc = personas.get(cluster, (f"Cluster {cluster}", "General customer segment."))
# # # # # # #     st.success(f"**Segment:** {label} (Cluster #{cluster})")
# # # # # # #     st.info(desc)
# # # # # # import streamlit as st
# # # # # # import pandas as pd
# # # # # # import joblib
# # # # # # import os

# # # # # # st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")

# # # # # # @st.cache_resource
# # # # # # def load_assets():
# # # # # #     base = os.path.dirname(__file__)
# # # # # #     # Fallback to local directory if __file__ pathing fails during deployment
# # # # # #     model_path = os.path.join(base, "model.joblib") if os.path.exists(os.path.join(base, "model.joblib")) else "model.joblib"
# # # # # #     scaler_path = os.path.join(base, "scaler.joblib") if os.path.exists(os.path.join(base, "scaler.joblib")) else "scaler.joblib"
    
# # # # # #     return joblib.load(model_path), joblib.load(scaler_path)

# # # # # # model, scaler = load_assets()

# # # # # # personas = {
# # # # # #     0: ("Balanced Spender", "Moderate income & moderate spending habits."),
# # # # # #     1: ("Conservative High-Earner", "High income but disciplined, low spending."),
# # # # # #     2: ("VIP / High Roller", "High income & high spending — prime target audience!"),
# # # # # #     3: ("Budget Conscious", "Low income & frugal, cautious spending."),
# # # # # #     4: ("Aspirational Spender", "Low income but passionate, high spending habits.")
# # # # # # }

# # # # # # st.title("🛍️ Advanced Customer Segmentation Predictor")
# # # # # # st.markdown("Predict customer marketing personas in real-time or via batch CSV uploads.")

# # # # # # # Create a tabbed interface
# # # # # # tab1, tab2, tab3 = st.tabs(["🎯 Real-Time Prediction", "📁 Batch Upload", "📊 Persona Insights"])

# # # # # # with tab1:
# # # # # #     st.subheader("Individual Customer Prediction")
    
# # # # # #     col1, col2 = st.columns(2)
# # # # # #     with col1:
# # # # # #         income = st.slider("Annual Income (k$)", min_value=10, max_value=150, value=60, step=1)
# # # # # #     with col2:
# # # # # #         spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)

# # # # # #     if st.button("Segment Customer", type="primary"):
# # # # # #         df = pd.DataFrame([[income, spending]], columns=["Annual Income (k$)", "Spending Score (1-100)"])
# # # # # #         cluster = int(model.predict(scaler.transform(df))[0])
# # # # # #         label, desc = personas.get(cluster, (f"Cluster {cluster}", "General customer segment."))
        
# # # # # #         st.success(f"**Segment:** {label} (Cluster #{cluster})")
# # # # # #         st.info(desc)

# # # # # # with tab2:
# # # # # #     st.subheader("Batch Customer Segmentation")
# # # # # #     st.write("Upload a CSV file containing customer data to segment an entire database at once.")
    
# # # # # #     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
# # # # # #     if uploaded_file is not None:
# # # # # #         batch_df = pd.read_csv(uploaded_file)
        
# # # # # #         # Check if required columns exist
# # # # # #         if "Annual Income (k$)" in batch_df.columns and "Spending Score (1-100)" in batch_df.columns:
# # # # # #             X_batch = batch_df[["Annual Income (k$)", "Spending Score (1-100)"]]
            
# # # # # #             # Predict and map labels
# # # # # #             predictions = model.predict(scaler.transform(X_batch))
# # # # # #             batch_df["Cluster"] = predictions
# # # # # #             batch_df["Persona"] = batch_df["Cluster"].map(lambda x: personas.get(x, ("Unknown",))[0])
            
# # # # # #             st.dataframe(batch_df, use_container_width=True)
            
# # # # # #             # Allow user to download the new dataset
# # # # # #             csv = batch_df.to_csv(index=False).encode('utf-8')
# # # # # #             st.download_button(
# # # # # #                 label="Download Segmented Data",
# # # # # #                 data=csv,
# # # # # #                 file_name="segmented_customers.csv",
# # # # # #                 mime="text/csv"
# # # # # #             )
# # # # # #         else:
# # # # # #             st.error("Error: CSV must contain 'Annual Income (k$)' and 'Spending Score (1-100)' columns.")

# # # # # # with tab3:
# # # # # #     st.subheader("Cluster Profiles")
# # # # # #     st.write("Understanding the 5 distinct customer cohorts discovered through our geometric K-Means partitioning[cite: 2]:")
    
# # # # # #     for cluster_id, (label, desc) in personas.items():
# # # # # #         st.markdown(f"**Cluster {cluster_id}: {label}**")
# # # # # #         st.write(f"- *Insight:* {desc}[cite: 2]")
# # # # # import streamlit as st
# # # # # import pandas as pd
# # # # # import joblib
# # # # # import os
# # # # # import plotly.express as px

# # # # # st.set_page_config(page_title="Customer Segmentation Pro", page_icon="🛍️", layout="wide")

# # # # # @st.cache_resource
# # # # # def load_assets():
# # # # #     base = os.path.dirname(__file__)
# # # # #     model_path = os.path.join(base, "model.joblib") if os.path.exists(os.path.join(base, "model.joblib")) else "model.joblib"
# # # # #     scaler_path = os.path.join(base, "scaler.joblib") if os.path.exists(os.path.join(base, "scaler.joblib")) else "scaler.joblib"
# # # # #     return joblib.load(model_path), joblib.load(scaler_path)

# # # # # model, scaler = load_assets()

# # # # # personas = {
# # # # #     0: ("Balanced Spender", "Moderate income & moderate spending habits."),
# # # # #     1: ("Conservative High-Earner", "High income but disciplined, low spending."),
# # # # #     2: ("VIP / High Roller", "High income & high spending — prime target audience!"),
# # # # #     3: ("Budget Conscious", "Low income & frugal, cautious spending."),
# # # # #     4: ("Aspirational Spender", "Low income but passionate, high spending habits.")
# # # # # }

# # # # # st.title("🛍️ Advanced Customer Segmentation Analytics")
# # # # # st.markdown("Predict customer marketing personas, analyze batch distributions, and visualize cluster centroids.")

# # # # # tab1, tab2, tab3, tab4 = st.tabs(["🎯 Real-Time Prediction", "📁 Batch Analytics", "📈 Cluster Visualization", "🧠 Model Explainability"])

# # # # # with tab1:
# # # # #     st.subheader("Individual Customer Prediction")
# # # # #     col1, col2 = st.columns(2)
# # # # #     with col1:
# # # # #         income = st.slider("Annual Income (k$)", min_value=10, max_value=150, value=60, step=1)
# # # # #     with col2:
# # # # #         spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)

# # # # #     if st.button("Segment Customer", type="primary"):
# # # # #         df = pd.DataFrame([[income, spending]], columns=["Annual Income (k$)", "Spending Score (1-100)"])
# # # # #         cluster = int(model.predict(scaler.transform(df))[0])
# # # # #         label, desc = personas.get(cluster, (f"Cluster {cluster}", "General customer segment."))
        
# # # # #         st.success(f"**Segment:** {label} (Cluster #{cluster})")
# # # # #         st.info(desc)

# # # # # with tab2:
# # # # #     st.subheader("Batch Customer Segmentation & Profiling")
# # # # #     uploaded_file = st.file_uploader("Upload CSV database", type=["csv"])
    
# # # # #     if uploaded_file is not None:
# # # # #         batch_df = pd.read_csv(uploaded_file)
        
# # # # #         if "Annual Income (k$)" in batch_df.columns and "Spending Score (1-100)" in batch_df.columns:
# # # # #             X_batch = batch_df[["Annual Income (k$)", "Spending Score (1-100)"]]
# # # # #             batch_df["Cluster"] = model.predict(scaler.transform(X_batch))
# # # # #             batch_df["Persona"] = batch_df["Cluster"].map(lambda x: personas.get(x, ("Unknown",))[0])
            
# # # # #             st.dataframe(batch_df, use_container_width=True)
            
# # # # #             st.markdown("**Cohort Demographics (Averages)**")
# # # # #             summary_df = batch_df.groupby("Persona")[["Annual Income (k$)", "Spending Score (1-100)"]].mean().round(2)
# # # # #             st.dataframe(summary_df, use_container_width=True)
            
# # # # #             csv = batch_df.to_csv(index=False).encode('utf-8')
# # # # #             st.download_button("Download Segmented Data", data=csv, file_name="segmented_customers.csv", mime="text/csv")
            
# # # # #             # Save to session state for the visualization tab
# # # # #             st.session_state['batch_df'] = batch_df
# # # # #         else:
# # # # #             st.error("CSV must contain 'Annual Income (k$)' and 'Spending Score (1-100)' columns.")

# # # # # with tab3:
# # # # #     st.subheader("Interactive Scatter Mapping")
# # # # #     if 'batch_df' in st.session_state:
# # # # #         df_plot = st.session_state['batch_df']
# # # # #         fig = px.scatter(
# # # # #             df_plot, 
# # # # #             x="Annual Income (k$)", 
# # # # #             y="Spending Score (1-100)", 
# # # # #             color="Persona",
# # # # #             title="Customer Segments Visualized",
# # # # #             hover_data=["Cluster"]
# # # # #         )
# # # # #         st.plotly_chart(fig, use_container_width=True)
# # # # #     else:
# # # # #         st.info("Upload a batch CSV in the 'Batch Analytics' tab to generate interactive visual maps.")

# # # # # with tab4:
# # # # #     st.subheader("K-Means Centroid Architecture")
# # # # #     st.write("Extracting the mathematical center of each customer cohort directly from the trained model.")
    
# # # # #     try:
# # # # #         # Inverse transform the centroids back to original scale for readability
# # # # #         centroids_scaled = model.cluster_centers_
# # # # #         centroids_original = scaler.inverse_transform(centroids_scaled)
        
# # # # #         centroid_df = pd.DataFrame(centroids_original, columns=["Annual Income (k$)", "Spending Score (1-100)"])
# # # # #         centroid_df.index.name = "Cluster ID"
# # # # #         centroid_df["Persona"] = centroid_df.index.map(lambda x: personas.get(x, ("Unknown",))[0])
        
# # # # #         st.dataframe(centroid_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
# # # # #     except AttributeError:
# # # # #         st.warning("Current model does not expose cluster_centers_. Ensure it is a standard Scikit-Learn KMeans object.")
# # # # import streamlit as st
# # # # import pandas as pd
# # # # import joblib
# # # # import os
# # # # import plotly.express as px

# # # # st.set_page_config(page_title="Customer Segmentation Pro", page_icon="🛍️", layout="wide")

# # # # @st.cache_resource
# # # # def load_assets():
# # # #     base = os.path.dirname(__file__)
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

# # # # st.title("🛍️ Advanced Customer Segmentation Analytics")
# # # # st.markdown("Predict customer marketing personas, analyze batch distributions, and visualize cluster centroids.")

# # # # tab1, tab2, tab3, tab4 = st.tabs(["🎯 Real-Time Prediction", "📁 Batch Analytics", "📈 Cluster Visualization", "🧠 Model Explainability"])

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
# # # #     st.subheader("Batch Customer Segmentation & Profiling")
# # # #     uploaded_file = st.file_uploader("Upload CSV database", type=["csv"])
    
# # # #     if uploaded_file is not None:
# # # #         batch_df = pd.read_csv(uploaded_file)
        
# # # #         if "Annual Income (k$)" in batch_df.columns and "Spending Score (1-100)" in batch_df.columns:
# # # #             X_batch = batch_df[["Annual Income (k$)", "Spending Score (1-100)"]]
# # # #             batch_df["Cluster"] = model.predict(scaler.transform(X_batch))
# # # #             batch_df["Persona"] = batch_df["Cluster"].map(lambda x: personas.get(x, ("Unknown",))[0])
            
# # # #             st.dataframe(batch_df, use_container_width=True)
            
# # # #             st.markdown("**Cohort Demographics (Averages)**")
# # # #             summary_df = batch_df.groupby("Persona")[["Annual Income (k$)", "Spending Score (1-100)"]].mean().round(2)
# # # #             st.dataframe(summary_df, use_container_width=True)
            
# # # #             csv = batch_df.to_csv(index=False).encode('utf-8')
# # # #             st.download_button("Download Segmented Data", data=csv, file_name="segmented_customers.csv", mime="text/csv")
            
# # # #             # Save to session state for the visualization tab
# # # #             st.session_state['batch_df'] = batch_df
# # # #         else:
# # # #             st.error("CSV must contain 'Annual Income (k$)' and 'Spending Score (1-100)' columns.")

# # # # with tab3:
# # # #     st.subheader("Interactive Scatter Mapping")
# # # #     if 'batch_df' in st.session_state:
# # # #         df_plot = st.session_state['batch_df']
# # # #         fig = px.scatter(
# # # #             df_plot, 
# # # #             x="Annual Income (k$)", 
# # # #             y="Spending Score (1-100)", 
# # # #             color="Persona",
# # # #             title="Customer Segments Visualized",
# # # #             hover_data=["Cluster"]
# # # #         )
# # # #         st.plotly_chart(fig, use_container_width=True)
# # # #     else:
# # # #         st.info("Upload a batch CSV in the 'Batch Analytics' tab to generate interactive visual maps.")

# # # # with tab4:
# # # #     st.subheader("K-Means Centroid Architecture")
# # # #     st.write("Extracting the mathematical center of each customer cohort directly from the trained model.")
    
# # # #     try:
# # # #         # Inverse transform the centroids back to original scale for readability
# # # #         centroids_scaled = model.cluster_centers_
# # # #         centroids_original = scaler.inverse_transform(centroids_scaled)
        
# # # #         centroid_df = pd.DataFrame(centroids_original, columns=["Annual Income (k$)", "Spending Score (1-100)"])
# # # #         centroid_df.index.name = "Cluster ID"
# # # #         centroid_df["Persona"] = centroid_df.index.map(lambda x: personas.get(x, ("Unknown",))[0])
        
# # # #         # Standard dataframe display without highlighting
# # # #         st.dataframe(centroid_df, use_container_width=True)
# # # #     except AttributeError:
# # # #         st.warning("Current model does not expose cluster_centers_. Ensure it is a standard Scikit-Learn KMeans object.")

# # # import streamlit as st
# # # import pandas as pd
# # # import joblib
# # # import os
# # # import plotly.express as px
# # # import time

# # # st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="📊", layout="wide")

# # # @st.cache_resource
# # # def load_assets():
# # #     base = os.path.dirname(__file__)
# # #     model_path = os.path.join(base, "model.joblib") if os.path.exists(os.path.join(base, "model.joblib")) else "model.joblib"
# # #     scaler_path = os.path.join(base, "scaler.joblib") if os.path.exists(os.path.join(base, "scaler.joblib")) else "scaler.joblib"
# # #     return joblib.load(model_path), joblib.load(scaler_path)

# # # model, scaler = load_assets()

# # # personas = {
# # #     0: ("Balanced Spender", "Moderate income & moderate spending habits. A stable, reliable customer base."),
# # #     1: ("Conservative High-Earner", "High income but disciplined, low spending. Prime target for premium, high-value, long-lasting products."),
# # #     2: ("VIP / High Roller", "High income & high spending. The most lucrative segment; target with exclusive offers and loyalty programs."),
# # #     3: ("Budget Conscious", "Low income & frugal, cautious spending. Respond well to discounts, sales, and essential goods."),
# # #     4: ("Aspirational Spender", "Low income but passionate, high spending habits. Potential targets for credit offers or trend-based marketing.")
# # # }

# # # # --- SIDEBAR: REAL-TIME PREDICTION ---
# # # with st.sidebar:
# # #     st.header("🎯 Quick Prediction")
# # #     st.markdown("Instantly segment a single customer.")
    
# # #     income = st.slider("Annual Income (k$)", 10, 150, 60, 1)
# # #     spending = st.slider("Spending Score (1-100)", 1, 100, 50, 1)

# # #     if st.button("Predict Segment", type="primary", use_container_width=True):
# # #         df = pd.DataFrame([[income, spending]], columns=["Annual Income (k$)", "Spending Score (1-100)"])
# # #         cluster = int(model.predict(scaler.transform(df))[0])
# # #         label, desc = personas.get(cluster, (f"Cluster {cluster}", "General segment."))
        
# # #         st.success(f"**{label}**")
# # #         st.caption(desc)

# # # # --- MAIN DASHBOARD ---
# # # st.title("📊 Customer Segmentation Dashboard")
# # # st.markdown("Analyze customer cohorts, visualize cluster distributions, and extract mathematical model insights.")

# # # tab1, tab2, tab3 = st.tabs(["📁 Batch Analytics", "📈 Interactive Visualization", "🧠 Model Architecture"])

# # # # TAB 1: BATCH ANALYTICS & METRICS
# # # with tab1:
# # #     st.subheader("Database Processing")
# # #     uploaded_file = st.file_uploader("Upload customer dataset (CSV)", type=["csv"])
    
# # #     if uploaded_file is not None:
# # #         with st.spinner("Processing dataset and running K-Means clustering..."):
# # #             time.sleep(0.5) # Simulate processing time for UX
# # #             batch_df = pd.read_csv(uploaded_file)
            
# # #             if "Annual Income (k$)" in batch_df.columns and "Spending Score (1-100)" in batch_df.columns:
# # #                 X_batch = batch_df[["Annual Income (k$)", "Spending Score (1-100)"]]
# # #                 batch_df["Cluster"] = model.predict(scaler.transform(X_batch))
# # #                 batch_df["Persona"] = batch_df["Cluster"].map(lambda x: personas.get(x, ("Unknown",))[0])
# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # from sklearn.cluster import KMeans
# # from sklearn.preprocessing import StandardScaler
# # import joblib
# # import os
# # import plotly.express as px
# # import time

# # st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="📊", layout="wide")

# # @st.cache_resource
# # def get_model_and_scaler():
# #     """Loads the model if it exists, otherwise trains and saves it automatically."""
# #     base_dir = os.path.dirname(__file__) if '__file__' in locals() else "."
# #     model_path = os.path.join(base_dir, "model.joblib")
# #     scaler_path = os.path.join(base_dir, "scaler.joblib")
    
# #     if os.path.exists(model_path) and os.path.exists(scaler_path):
# #         return joblib.load(model_path), joblib.load(scaler_path)
    
# #     # Train the model if files are missing
# #     np.random.seed(42)
# #     data = {
# #         "Annual Income (k$)": np.random.randint(15, 140, 200),
# #         "Spending Score (1-100)": np.random.randint(1, 100, 200)
# #     }
# #     df = pd.DataFrame(data)
    
# #     scaler = StandardScaler()
# #     X_scaled = scaler.fit_transform(df)
    
# #     kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
# #     kmeans.fit(X_scaled)
    
# #     # Save for future runs
# #     joblib.dump(kmeans, model_path)
# #     joblib.dump(scaler, scaler_path)
    
# #     return kmeans, scaler

# # # Initialize model and scaler
# # model, scaler = get_model_and_scaler()

# # personas = {
# #     0: ("Balanced Spender", "Moderate income & moderate spending habits. A stable, reliable customer base."),
# #     1: ("Conservative High-Earner", "High income but disciplined, low spending. Prime target for premium, high-value, long-lasting products."),
# #     2: ("VIP / High Roller", "High income & high spending. The most lucrative segment; target with exclusive offers and loyalty programs."),
# #     3: ("Budget Conscious", "Low income & frugal, cautious spending. Respond well to discounts, sales, and essential goods."),
# #     4: ("Aspirational Spender", "Low income but passionate, high spending habits. Potential targets for credit offers or trend-based marketing.")
# # }

# # # --- SIDEBAR: REAL-TIME PREDICTION ---
# # with st.sidebar:
# #     st.header("🎯 Quick Prediction")
# #     st.markdown("Instantly segment a single customer.")
    
# #     income = st.slider("Annual Income (k$)", 10, 150, 60, 1)
# #     spending = st.slider("Spending Score (1-100)", 1, 100, 50, 1)

# #     if st.button("Predict Segment", type="primary", use_container_width=True):
# #         df = pd.DataFrame([[income, spending]], columns=["Annual Income (k$)", "Spending Score (1-100)"])
# #         cluster = int(model.predict(scaler.transform(df))[0])
# #         label, desc = personas.get(cluster, (f"Cluster {cluster}", "General segment."))
        
# #         st.success(f"**{label}**")
# #         st.caption(desc)

# # # --- MAIN DASHBOARD ---
# # st.title("📊 Customer Segmentation Dashboard")
# # st.markdown("Analyze customer cohorts, visualize cluster distributions, and extract mathematical model insights.")

# # tab1, tab2, tab3 = st.tabs(["📁 Batch Analytics", "📈 Interactive Visualization", "🧠 Model Architecture"])

# # # TAB 1: BATCH ANALYTICS & METRICS
# # with tab1:
# #     st.subheader("Database Processing")
# #     uploaded_file = st.file_uploader("Upload customer dataset (CSV)", type=["csv"])
    
# #     if uploaded_file is not None:
# #         with st.spinner("Processing dataset and running K-Means clustering..."):
# #             time.sleep(0.5)
# #             batch_df = pd.read_csv(uploaded_file)
            
# #             if "Annual Income (k$)" in batch_df.columns and "Spending Score (1-100)" in batch_df.columns:
# #                 X_batch = batch_df[["Annual Income (k$)", "Spending Score (1-100)"]]
                
# # #                 # Dynamic KPI Metrics
# # #                 col1, col2, col3 = st.columns(3)
# # #                 col1.metric("Total Customers Processed", len(batch_df))
# # #                 top_segment = batch_df["Persona"].mode
# import streamlit as st
# import pandas as pd
# import joblib
# import os
# import plotly.express as px

# st.set_page_config(page_title="Customer Segmentation Pro", page_icon="🛍️", layout="wide")

# @st.cache_resource
# def load_assets():
#     base = os.path.dirname(__file__)
#     model_path = os.path.join(base, "model.joblib") if os.path.exists(os.path.join(base, "model.joblib")) else "model.joblib"
#     scaler_path = os.path.join(base, "scaler.joblib") if os.path.exists(os.path.join(base, "scaler.joblib")) else "scaler.joblib"
#     return joblib.load(model_path), joblib.load(scaler_path)

# model, scaler = load_assets()

# personas = {
#     0: ("Balanced Spender", "Moderate income & moderate spending habits."),
#     1: ("Conservative High-Earner", "High income but disciplined, low spending."),
#     2: ("VIP / High Roller", "High income & high spending — prime target audience!"),
#     3: ("Budget Conscious", "Low income & frugal, cautious spending."),
#     4: ("Aspirational Spender", "Low income but passionate, high spending habits.")
# }

# st.title("🛍️ Advanced Customer Segmentation Analytics")
# st.markdown("Predict customer marketing personas, analyze batch distributions, and visualize cluster centroids.")

# tab1, tab2, tab3, tab4 = st.tabs(["🎯 Real-Time Prediction", "📁 Batch Analytics", "📈 Cluster Visualization", "🧠 Model Explainability"])

# with tab1:
#     st.subheader("Individual Customer Prediction")
#     col1, col2 = st.columns(2)
#     with col1:
#         income = st.slider("Annual Income (k$)", min_value=10, max_value=150, value=60, step=1)
#     with col2:
#         spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)

#     if st.button("Segment Customer", type="primary"):
#         df = pd.DataFrame([[income, spending]], columns=["Annual Income (k$)", "Spending Score (1-100)"])
#         cluster = int(model.predict(scaler.transform(df))[0])
#         label, desc = personas.get(cluster, (f"Cluster {cluster}", "General customer segment."))
        
#         st.success(f"**Segment:** {label} (Cluster #{cluster})")
#         st.info(desc)

# with tab2:
#     st.subheader("Batch Customer Segmentation & Profiling")
#     uploaded_file = st.file_uploader("Upload CSV database", type=["csv"])
    
#     if uploaded_file is not None:
#         batch_df = pd.read_csv(uploaded_file)
        
#         if "Annual Income (k$)" in batch_df.columns and "Spending Score (1-100)" in batch_df.columns:
#             X_batch = batch_df[["Annual Income (k$)", "Spending Score (1-100)"]]
#             batch_df["Cluster"] = model.predict(scaler.transform(X_batch))
#             batch_df["Persona"] = batch_df["Cluster"].map(lambda x: personas.get(x, ("Unknown",))[0])
            
#             st.dataframe(batch_df, use_container_width=True)
            
#             st.markdown("**Cohort Demographics (Averages)**")
#             summary_df = batch_df.groupby("Persona")[["Annual Income (k$)", "Spending Score (1-100)"]].mean().round(2)
#             st.dataframe(summary_df, use_container_width=True)
            
#             csv = batch_df.to_csv(index=False).encode('utf-8')
#             st.download_button("Download Segmented Data", data=csv, file_name="segmented_customers.csv", mime="text/csv")
            
#             # Save to session state for the visualization tab
#             st.session_state['batch_df'] = batch_df
#         else:
#             st.error("CSV must contain 'Annual Income (k$)' and 'Spending Score (1-100)' columns.")

# with tab3:
#     st.subheader("Interactive Scatter Mapping")
#     if 'batch_df' in st.session_state:
#         df_plot = st.session_state['batch_df']
#         fig = px.scatter(
#             df_plot, 
#             x="Annual Income (k$)", 
#             y="Spending Score (1-100)", 
#             color="Persona",
#             title="Customer Segments Visualized",
#             hover_data=["Cluster"]
#         )
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info("Upload a batch CSV in the 'Batch Analytics' tab to generate interactive visual maps.")

# with tab4:
#     st.subheader("K-Means Centroid Architecture")
#     st.write("Extracting the mathematical center of each customer cohort directly from the trained model.")
    
#     try:
#         # Inverse transform the centroids back to original scale for readability
#         centroids_scaled = model.cluster_centers_
#         centroids_original = scaler.inverse_transform(centroids_scaled)
        
#         centroid_df = pd.DataFrame(centroids_original, columns=["Annual Income (k$)", "Spending Score (1-100)"])
#         centroid_df.index.name = "Cluster ID"
#         centroid_df["Persona"] = centroid_df.index.map(lambda x: personas.get(x, ("Unknown",))[0])
        
#         st.dataframe(centroid_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
#     except AttributeError:
#         st.warning("Current model does not expose cluster_centers_. Ensure it is a standard Scikit-Learn KMeans object.")

import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linear_sum_assignment

INCOME = "Annual Income (k$)"
SPEND = "Spending Score (1-100)"
FEATURES = [INCOME, SPEND]

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")

# ----------------------------------------------------------------------------
# Styling (kept light; the base look comes from .streamlit/config.toml)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; max-width: 1200px;}
    h1 {font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.2rem;}
    .subtitle {color: #5B6478; font-size: 1.05rem; margin-bottom: 1.4rem;}

    div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E3E7F0; border-radius: 12px;
        padding: 14px 18px;
    }
    div[data-testid="stMetricLabel"] p {color: #5B6478;}

    button[data-baseweb="tab"] {font-size: 1rem; font-weight: 600; padding: 10px 18px;}

    .result-card {
        background: #FFFFFF; border: 1px solid #E3E7F0; border-left: 8px solid var(--c);
        border-radius: 12px; padding: 22px 26px; margin-top: 4px;
    }
    .result-card .name {font-size: 1.7rem; font-weight: 800; color: var(--c); line-height: 1.2;}
    .result-card .desc {color: #3A4256; margin-top: 6px; font-size: 1.02rem;}
    .result-card .meta {color: #5B6478; margin-top: 14px; font-size: 0.92rem;}

    .legend-row {display: flex; align-items: flex-start; gap: 10px; margin-bottom: 12px;}
    .legend-dot {width: 12px; height: 12px; border-radius: 50%; margin-top: 6px; flex-shrink: 0;}
    .legend-name {font-weight: 700; font-size: 0.95rem;}
    .legend-desc {color: #5B6478; font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Model loading (unchanged logic)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    base = os.path.dirname(__file__)
    model_path = os.path.join(base, "model.joblib") if os.path.exists(os.path.join(base, "model.joblib")) else "model.joblib"
    scaler_path = os.path.join(base, "scaler.joblib") if os.path.exists(os.path.join(base, "scaler.joblib")) else "scaler.joblib"
    return joblib.load(model_path), joblib.load(scaler_path)


try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error(
        "model.joblib / scaler.joblib not found. Run `python train_model.py` once "
        "and commit both files next to app.py."
    )
    st.stop()


def predict(df: pd.DataFrame) -> np.ndarray:
    """Same pipeline as before: scale, then KMeans predict."""
    return model.predict(scaler.transform(df[FEATURES]))


# ----------------------------------------------------------------------------
# Personas
# KMeans cluster IDs are arbitrary (cluster 0 is not always "Balanced Spender"),
# so we match each trained centroid to the closest persona archetype instead of
# hard-coding ID -> name. The model itself is untouched.
# ----------------------------------------------------------------------------
PERSONA_DEFS = {
    "Balanced Spender": ("Moderate income & moderate spending habits.", "#3D6FB6", (55, 50)),
    "Conservative High-Earner": ("High income but disciplined, low spending.", "#2E9E8F", (88, 17)),
    "VIP / High Roller": ("High income & high spending. Prime target audience.", "#E0A100", (88, 82)),
    "Budget Conscious": ("Low income & frugal, cautious spending.", "#7A8599", (26, 20)),
    "Aspirational Spender": ("Low income but passionate, high spending habits.", "#D6476B", (26, 79)),
}
FALLBACK_COLORS = px.colors.qualitative.Safe


@st.cache_data
def build_personas():
    n = len(model.cluster_centers_)
    if n == len(PERSONA_DEFS):
        names = list(PERSONA_DEFS)
        archetypes = pd.DataFrame([PERSONA_DEFS[k][2] for k in names], columns=FEATURES)
        arche_scaled = scaler.transform(archetypes)
        cost = np.linalg.norm(model.cluster_centers_[:, None, :] - arche_scaled[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        return {int(r): (names[c], PERSONA_DEFS[names[c]][0], PERSONA_DEFS[names[c]][1]) for r, c in zip(rows, cols)}
    return {i: (f"Segment {i + 1}", "General customer segment.", FALLBACK_COLORS[i % len(FALLBACK_COLORS)]) for i in range(n)}


personas = build_personas()
name_of = lambda c: personas.get(int(c), (f"Cluster {c}",))[0]
color_map = {v[0]: v[2] for v in personas.values()}
centroids = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=FEATURES)
centroids.index.name = "Cluster ID"
centroids["Persona"] = [name_of(i) for i in centroids.index]


def style_fig(fig, height=460):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, title=None),
    )
    fig.update_xaxes(gridcolor="#EEF1F6", zeroline=False)
    fig.update_yaxes(gridcolor="#EEF1F6", zeroline=False)
    return fig


def centroid_trace():
    return go.Scatter(
        x=centroids[INCOME], y=centroids[SPEND], mode="markers+text",
        text=centroids["Persona"], textposition="top center", name="Segment centre",
        marker=dict(size=16, symbol="diamond", color=[color_map[p] for p in centroids["Persona"]],
                    line=dict(width=2, color="#1B1F3B")),
        hovertemplate="%{text}<br>Income %{x:.0f}k, Spend %{y:.0f}<extra></extra>",
    )


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Segments")
    for cid in sorted(personas):
        n, d, c = personas[cid]
        st.markdown(
            f'<div class="legend-row"><div class="legend-dot" style="background:{c}"></div>'
            f'<div><div class="legend-name">{n}</div><div class="legend-desc">{d}</div></div></div>',
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption(f"Model: K-Means, {len(personas)} clusters, trained on income and spending score (scaled).")

    rng = np.random.default_rng(7)
    sample = pd.DataFrame({INCOME: rng.integers(15, 140, 60), SPEND: rng.integers(1, 100, 60)})
    st.download_button("Download sample CSV", sample.to_csv(index=False).encode("utf-8"),
                       file_name="sample_customers.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🛍️ Customer Segmentation")
st.markdown(
    '<div class="subtitle">Find out which marketing segment a customer belongs to, '
    "one at a time or for a whole CSV.</div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Batch upload", "Cluster map", "How it works"])

# ----------------------------------------------------------------------------
# Tab 1: single prediction
# ----------------------------------------------------------------------------
with tab1:
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.subheader("Customer details")
        income = st.slider(INCOME, min_value=10, max_value=150, value=60, step=1)
        spending = st.slider(SPEND, min_value=1, max_value=100, value=50, step=1)
        run = st.button("Segment customer", type="primary", use_container_width=True)

        if run:
            st.session_state["single"] = (income, spending)

    with right:
        if "single" in st.session_state:
            inc, spd = st.session_state["single"]
            df = pd.DataFrame([[inc, spd]], columns=FEATURES)
            cluster = int(predict(df)[0])
            label, desc, color = personas.get(cluster, (f"Cluster {cluster}", "General customer segment.", "#3D6FB6"))

            runner_up = ""
            try:
                dists = model.transform(scaler.transform(df))[0]
                order = np.argsort(dists)
                if len(order) > 1:
                    runner_up = f"Next closest segment: {name_of(order[1])}"
            except Exception:
                pass

            st.markdown(
                f'<div class="result-card" style="--c:{color}">'
                f'<div class="name">{label}</div><div class="desc">{desc}</div>'
                f'<div class="meta">Cluster #{cluster} &nbsp;|&nbsp; Income {inc}k &nbsp;|&nbsp; Score {spd}'
                f'{" &nbsp;|&nbsp; " + runner_up if runner_up else ""}</div></div>',
                unsafe_allow_html=True,
            )

            fig = go.Figure()
            fig.add_trace(centroid_trace())
            fig.add_trace(go.Scatter(
                x=[inc], y=[spd], mode="markers", name="This customer",
                marker=dict(size=18, symbol="star", color="#1B1F3B", line=dict(width=1, color="white")),
            ))
            fig.update_layout(title="Where this customer sits", xaxis_title=INCOME, yaxis_title=SPEND)
            st.plotly_chart(style_fig(fig, 380))
        else:
            st.info("Set the two sliders and press **Segment customer** to see the result.")

# ----------------------------------------------------------------------------
# Tab 2: batch
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Segment a customer file")
    st.caption(f"Your CSV needs two columns: `{INCOME}` and `{SPEND}`. Extra columns are kept.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read that file as CSV: {e}")
            raw = None

        if raw is not None:
            if not all(c in raw.columns for c in FEATURES):
                missing = [c for c in FEATURES if c not in raw.columns]
                st.error(f"Missing column(s): {', '.join(missing)}. Rename them to match exactly and upload again.")
            else:
                batch_df = raw.copy()
                batch_df[FEATURES] = batch_df[FEATURES].apply(pd.to_numeric, errors="coerce")
                bad = batch_df[FEATURES].isna().any(axis=1).sum()
                if bad:
                    st.warning(f"Skipped {bad} row(s) with empty or non-numeric values.")
                    batch_df = batch_df.dropna(subset=FEATURES)

                if batch_df.empty:
                    st.error("No valid rows left to segment.")
                else:
                    batch_df["Cluster"] = predict(batch_df)
                    batch_df["Persona"] = batch_df["Cluster"].map(name_of)
                    st.session_state["batch_df"] = batch_df

    if "batch_df" in st.session_state:
        b = st.session_state["batch_df"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Customers", f"{len(b):,}")
        m2.metric("Segments found", b["Persona"].nunique())
        m3.metric("Avg income", f"{b[INCOME].mean():.1f}k")
        m4.metric("Avg spending score", f"{b[SPEND].mean():.1f}")

        c1, c2 = st.columns([1, 1.2], gap="large")
        counts = b["Persona"].value_counts().reset_index()
        counts.columns = ["Persona", "Customers"]
        with c1:
            fig = px.pie(counts, names="Persona", values="Customers", hole=0.55,
                         color="Persona", color_discrete_map=color_map, title="Segment share")
            fig.update_traces(textinfo="percent")
            st.plotly_chart(style_fig(fig, 340))
        with c2:
            summary = (b.groupby("Persona")[FEATURES].mean().round(2)
                       .join(b["Persona"].value_counts().rename("Customers")).sort_values("Customers", ascending=False))
            st.markdown("**Segment averages**")
            st.dataframe(summary)

        st.markdown("**All customers**")
        persona_filter = st.multiselect("Filter by segment", sorted(b["Persona"].unique()))
        view = b[b["Persona"].isin(persona_filter)] if persona_filter else b
        st.dataframe(view, hide_index=True)

        st.download_button("Download segmented data", b.to_csv(index=False).encode("utf-8"),
                           file_name="segmented_customers.csv", mime="text/csv", type="primary")
    else:
        st.info("Upload a CSV above. Need a file to try? Grab the sample from the sidebar.")

# ----------------------------------------------------------------------------
# Tab 3: cluster map
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Cluster map")
    fig = go.Figure()
    if "batch_df" in st.session_state:
        df_plot = st.session_state["batch_df"]
        for p, g in df_plot.groupby("Persona"):
            fig.add_trace(go.Scatter(
                x=g[INCOME], y=g[SPEND], mode="markers", name=p,
                marker=dict(size=8, color=color_map.get(p), opacity=0.75),
                hovertemplate=f"{p}<br>Income %{{x}}k, Spend %{{y}}<extra></extra>",
            ))
    else:
        st.info("Showing segment centres only. Upload a CSV in **Batch upload** to see your customers on the map.")
    fig.add_trace(centroid_trace())
    fig.update_layout(xaxis_title=INCOME, yaxis_title=SPEND)
    st.plotly_chart(style_fig(fig, 540))

# ----------------------------------------------------------------------------
# Tab 4: explainability
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("How the model decides")
    st.markdown(
        "K-Means splits customers into groups around a **centre point** per group. "
        "A new customer is put into the group whose centre is closest. "
        "Income and spending score are scaled first so neither one dominates the distance."
    )

    st.markdown("**Segment centres (original units)**")
    try:
        st.dataframe(centroids.style.highlight_max(axis=0, subset=FEATURES, color="#CFEFD9").format({INCOME: "{:.1f}", SPEND: "{:.1f}"}))
    except Exception:
        st.dataframe(centroids)

    fig = px.bar(centroids.melt(id_vars="Persona", value_vars=FEATURES, var_name="Feature", value_name="Value"),
                 x="Persona", y="Value", color="Feature", barmode="group",
                 color_discrete_sequence=["#3D6FB6", "#D6476B"], title="Centre of each segment")
    fig.update_layout(xaxis_title=None)
    st.plotly_chart(style_fig(fig, 380))
