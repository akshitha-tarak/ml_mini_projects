import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

np.random.seed(42)
df = pd.DataFrame({
    "Annual Income (k$)": np.random.randint(15, 140, 200),
    "Spending Score (1-100)": np.random.randint(1, 100, 200)
})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
kmeans.fit(X_scaled)

joblib.dump(kmeans, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
