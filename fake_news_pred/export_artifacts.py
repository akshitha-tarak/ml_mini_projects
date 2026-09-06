import os
import sys
import re

_venv_site = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "Lib", "site-packages"))
if os.path.exists(_venv_site) and _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"[^a-z0-9\s\.\,\!\?\-\']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def export():
    csv_path = os.path.join(os.path.dirname(__file__), "WELFake_Dataset.csv")
    print(f"Loading dataset from {csv_path}...")
    # Use balanced sample matching notebook (10,000 real, 10,000 fake)
    data = pd.read_csv(csv_path)
    data = data.dropna(subset=["title", "text", "label"])
    
    df_fake = data[data["label"] == 0].sample(10000, random_state=42)
    df_real = data[data["label"] == 1].sample(10000, random_state=42)
    df = pd.concat([df_fake, df_real]).sample(frac=1, random_state=42)
    
    print("Preprocessing text...")
    content = (df["title"] + " " + df["text"]).apply(clean_text)
    y = df["label"].astype(int)
    
    print("Training TF-IDF + LogisticRegression pipeline...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=35000, ngram_range=(1, 2), stop_words="english", min_df=2)),
        ("model", LogisticRegression(max_iter=300, random_state=42))
    ])
    pipeline.fit(content, y)
    
    out_path = os.path.join(os.path.dirname(__file__), "model_pipeline.joblib")
    joblib.dump(pipeline, out_path, compress=3)
    print(f"Pipeline saved to: {out_path} (Size: {os.path.getsize(out_path)/(1024*1024):.2f} MB)")

if __name__ == "__main__":
    export()
