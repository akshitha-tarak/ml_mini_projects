# 🚀 Machine Learning Mini-Projects (`ml-mini-projects`)

A curated monorepo containing baseline machine learning models, exploratory data analyses, and deployment-ready pipelines across core machine learning domains: **Classification**, **Clustering**, **Natural Language Processing (NLP)**, and **Time Series Forecasting**.

---

## 📌 Repository Strategy

### Why a Monorepo for Baseline Projects?
Publishing separate repositories for standalone single-notebook experiments clutters your GitHub profile and dilutes your work. Consolidating baseline projects into **`ml-mini-projects`** offers several distinct advantages:
- **Clean, Unified Portfolio**: Presents diverse machine learning capabilities (tabular, text, time series, unsupervised) under a single cohesive, well-documented roof.
- **Consistent Code Standards**: Enforces standardized dependency management (`requirements.txt`), artifact serialization (`joblib`), and deployment templates across all projects.
- **Side-by-Side Benchmarkability**: Enables immediate comparison of architectures, loss functions, scaling techniques, and evaluation metrics across problem domains.

### 🔄 When to Spin Out into Standalone Repositories
Spin out an individual project from this monorepo into its own dedicated repository **only when scaling it into a full-stack, production ML system**, which includes:
- [ ] **Automated Data Ingestion**: Scheduled ETL/ELT pipelines with data validation (e.g., Great Expectations, dbt, Apache Airflow).
- [ ] **Production API & Microservice**: Fast serving layer using FastAPI / Docker containerization with request validation.
- [ ] **Comprehensive Test Suite**: Automated unit and integration testing (pytest, data drift detection).
- [ ] **CI/CD & MLOps Pipelines**: Automated model evaluation, GitHub Actions CI/CD, and experiment tracking (e.g., MLflow, Weights & Biases).
- [ ] **Custom Full-Stack UI**: A dedicated React / Next.js / Flutter frontend interface.

---

## 📊 Master Project Summary

| Project | Task Type | Primary Algorithm | Key Evaluation Metrics | Interactive App | Direct Link |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **[Credit Card Fraud Detection](./Credit_Card_Fraud)** | Imbalanced Binary Classification | Random Forest (Balanced Subsample) / Decision Tree | **Test Acc:** 99.95%<br>**PR-AUC:** 0.8537<br>Balanced class reweighting | ✅ Streamlit | [Explore Folder](./Credit_Card_Fraud) |
| **[Sonar Rock vs. Mine Classifier](./Sonar_Classification)** | Binary Classification (Signal Analysis) | Support Vector Classifier (`SVC`) + `StandardScaler` | **Test Acc:** 92.86%<br>**Train Acc:** 96.99% | ✅ Streamlit | [Explore Folder](./Sonar_Classification) |
| **[Customer Segmentation](./customer_segmentation)** | Unsupervised Clustering | K-Means ($k=5$) + `StandardScaler` | **Elbow Method:** Optimal $k=5$<br>**Silhouette Analysis** | ✅ Streamlit | [Explore Folder](./customer_segmentation) |
| **[Fake News Prediction](./fake_news_pred)** | NLP Binary Text Classification | `TfidfVectorizer` + Logistic Regression Pipeline | **Test Acc:** 93.48%<br>**F1-Score:** 0.93<br>35,000 n-gram features | ✅ Streamlit | [Explore Folder](./fake_news_pred) |
| **[Airline Passenger Forecasting](./time-series)** | Time Series Forecasting | SARIMAX $(1,1,1)\times(1,1,1,12)$ with Log Transform | **MSE:** 277.41<br>**MAE:** 11.99<br>**RMSE:** 16.66 | 📈 Forecast | [Explore Folder](./time-series) |

---

## 🛠️ Zero-Pickle Deployment Pipeline

A Jupyter Notebook alone cannot serve real-time predictions efficiently in production. To transition from an exploratory notebook to a live, production-grade service, follow this 4-step pipeline:

```
┌─────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐       ┌────────────────────────┐
│ Jupyter Notebook│ ───>  │  joblib Serialization │ ───>  │ Streamlit Frontend   │ ───>  │ Streamlit Cloud Deploy │
│ (Model Training)│       │ (model.joblib, etc.) │       │ (app.py < 40 lines)  │       │ (Free Live Web App)    │
└─────────────────┘       └──────────────────────┘       └──────────────────────┘       └────────────────────────┘
```

### 1. Export Trained Artifacts with `joblib`
Avoid standard Python `pickle` files for model serialization due to security vulnerabilities (arbitrary code execution) and compatibility issues. Instead, export the fitted model and preprocessing transformers using `joblib`:

```python
import joblib

# Export fitted model and any feature scalers or vectorizers
joblib.dump(best_model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
```

### 2. Build a Lightweight Frontend (`app.py`)
Build an interactive UI in fewer than 40 lines of clean code using [Streamlit](https://streamlit.io/). Load the `.joblib` artifacts, capture inputs through sliders or forms, and return real-time predictions:

```python
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️")
model = joblib.load("customer_segmentation/model.joblib")
scaler = joblib.load("customer_segmentation/scaler.joblib")

st.title("🛍️ Customer Segmentation Predictor")
income = st.slider("Annual Income (k$)", 10, 150, 60)
spending = st.slider("Spending Score (1-100)", 1, 100, 50)

if st.button("Predict Segment", type="primary"):
    data = pd.DataFrame([[income, spending]], columns=["Annual Income (k$)", "Spending Score (1-100)"])
    cluster = int(model.predict(scaler.transform(data))[0])
    st.success(f"Assigned Customer Cluster: **#{cluster}**")
```

### 3. Specify Dependencies (`requirements.txt`)
List runtime packages with pinned versions to ensure deterministic and reproducible deployments:

```txt
streamlit>=1.35.0
scikit-learn>=1.4.0
pandas>=2.2.0
numpy>=1.26.0
joblib>=1.4.0
statsmodels>=0.14.0
```

### 4. Host for Free on Streamlit Community Cloud
1. Push your repository to **GitHub** (with `.gitignore` configured to exclude large raw CSVs over 100MB).
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **"Create app"**, select your `ml-mini-projects` repository, set the branch to `main`, and specify the app path (e.g., `customer_segmentation/app.py`).
4. Click **"Deploy"** to launch your live application with a public URL.

---

## 📂 Repository Structure

```
ml-mini-projects/
│
├── README.md                           # Master repository documentation & portfolio index
├── requirements.txt                    # Pinned runtime dependencies
├── .gitignore                          # Excludes virtual environments and large raw CSVs (>100MB)
│
├── Credit_Card_Fraud/                  # Imbalanced Classification
│   ├── credit.ipynb                    # EDA, class imbalance handling, Random Forest
│   ├── export_artifacts.py             # Script to train & export model.joblib
│   ├── app.py                          # Streamlit real-time fraud risk evaluator (< 35 lines)
│   ├── model.joblib                    # Balanced Random Forest classifier artifact
│   └── creditcard.csv                  # Dataset (ignored in Git if >100MB)
│
├── Sonar_Classification/               # Binary Signal Classification
│   ├── sonar.ipynb                     # EDA, outlier analysis, SVC training & evaluation
│   ├── export_artifacts.py             # Script to export model.joblib & scaler.joblib
│   ├── app.py                          # Streamlit real-time signal classification UI (< 35 lines)
│   ├── model.joblib                    # Trained Support Vector Classifier
│   ├── scaler.joblib                   # Fitted StandardScaler
│   └── sonardata.csv                   # Sonar reflection dataset
│
├── customer_segmentation/              # Unsupervised Customer Segmentation
│   ├── custo_seg.ipynb                 # EDA, Elbow method, Silhouette analysis, K-Means
│   ├── export_artifacts.py             # Script to export model.joblib & scaler.joblib
│   ├── app.py                          # Streamlit customer segment predictor (< 40 lines)
│   ├── model.joblib                    # Trained KMeans (k=5) cluster model
│   ├── scaler.joblib                   # Fitted StandardScaler
│   └── Mall_Customers.csv              # Customer demographic & spending dataset
│
├── fake_news_pred/                     # NLP Text Classification
│   ├── fake_news.ipynb                 # Text cleaning, TF-IDF vectorization, Logistic Regression
│   ├── export_artifacts.py             # Script to train & export model_pipeline.joblib
│   ├── app.py                          # Streamlit news credibility evaluator (< 35 lines)
│   ├── model_pipeline.joblib           # Trained TF-IDF + Logistic Regression pipeline
│   └── WELFake_Dataset.csv             # Dataset (ignored in Git if >100MB)
│
└── time-series/                        # Time Series Analysis & Forecasting
    ├── timefore.ipynb                  # Trend/seasonality decomposition, log-transform, SARIMAX
    └── airline-passengers.csv          # International airline passenger dataset
```

---

## ⚡ Quickstart: Running Locally

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/<your-username>/ml-mini-projects.git
cd ml-mini-projects
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch Interactive Streamlit Apps
- **Customer Segmentation**:
  ```bash
  streamlit run customer_segmentation/app.py
  ```
- **Sonar Signal Classifier**:
  ```bash
  streamlit run Sonar_Classification/app.py
  ```
- **Fake News Verifier**:
  ```bash
  streamlit run fake_news_pred/app.py
  ```
- **Credit Card Fraud Detection**:
  ```bash
  streamlit run Credit_Card_Fraud/app.py
  ```

---

## 📜 License
This repository is open source and available under the [MIT License](LICENSE).
