import streamlit as st
import joblib, os, re

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

@st.cache_resource
def load_pipeline():
    base = os.path.dirname(__file__)
    return joblib.load(os.path.join(base, "model_pipeline.joblib"))

def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"[^a-z0-9\s\.\,\!\?\-\']", " ", s)
    return re.sub(r"\s+", " ", s).strip()

st.title("📰 Fake News Detection System")
st.write("Evaluate news veracity in real time using a TF-IDF & Logistic Regression NLP pipeline.")

title = st.text_input("News Article Title", "NASA's James Webb Telescope Discovers New Distant Galaxy")
text = st.text_area("Article Body Text", "Astronomers using NASA's James Webb Space Telescope have identified a distant galaxy that formed shortly after the Big Bang, according to peer-reviewed findings published this week.", height=120)

if st.button("Analyze Credibility", type="primary"):
    if not (title.strip() or text.strip()):
        st.warning("Please enter a title or body text to analyze.")
    else:
        pipeline = load_pipeline()
        cleaned = clean_text(f"{title} {text}")
        pred = pipeline.predict([cleaned])[0]
        prob = pipeline.predict_proba([cleaned])[0]
        if pred == 1:
            st.success(f"✅ **Verified Authentic / Real News** (Confidence: {prob[1]:.1%})")
        else:
            st.error(f"🚨 **High Risk: Likely Fake News** (Confidence: {prob[0]:.1%})")
