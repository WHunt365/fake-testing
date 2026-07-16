import streamlit as st
import requests
from bs4 import BeautifulSoup
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

MODEL_FILE = 'fake_news_rf_model.joblib'

# --- 1. HELPER FUNCTIONS ---

@st.cache_resource
def load_or_train_model():
    """Loads the model if it exists, otherwise trains a basic dummy model."""
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    
    # Train a basic dummy model if none exists
    data = {
        'text': [
            "The earth is definitely flat, a new study by anonymous scientists claims.",
            "The stock market saw a 2% increase today following the federal reserve report.",
            "Aliens have landed in Central Park and are handing out free smartphones.",
            "The local city council voted 5-2 to increase funding for public libraries."
        ],
        'label': ["Fake", "Real", "Fake", "Real"]
    }
    df = pd.DataFrame(data)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    pipeline.fit(df['text'], df['label'])
    joblib.dump(pipeline, MODEL_FILE)
    return pipeline

def scrape_article_text(url):
    """Scrapes paragraph text from a given news article URL."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text() for p in paragraphs]).strip()
    except Exception as e:
        return None

# --- 2. STREAMLIT UI ---

st.set_page_config(page_title="Fake News Detector", page_icon="🕵️‍♂️")

st.title("🕵️‍♂️ Fake News AI Detector")
st.markdown("""
Welcome to the Fake News Detector! Paste the URL of a news article below, 
and our Random Forest Machine Learning model will predict the likelihood of it being real or fake.
""")

# Load the model into memory
model = load_or_train_model()

# User Input
url_input = st.text_input("Paste News Article URL here:", placeholder="https://www.example.com/news-article")

if st.button("Analyze Article", type="primary"):
    if not url_input:
        st.warning("Please enter a URL first!")
    else:
        with st.spinner("Scraping article and analyzing text..."):
            article_text = scrape_article_text(url_input)
            
            if not article_text or len(article_text) < 50:
                st.error("Could not extract enough text from this URL. The site might be blocking web scrapers.")
            else:
                st.success(f"Successfully extracted {len(article_text)} characters of text!")
                
                # Make Prediction
                classes = model.classes_
                probabilities = model.predict_proba([article_text])[0]
                prob_dict = dict(zip(classes, probabilities))
                
                fake_percent = prob_dict.get('Fake', 0.0) * 100
                real_percent = prob_dict.get('Real', 0.0) * 100
                
                # Display Results beautifully
                st.markdown("### Detection Results")
                
                col1, col2 = st.columns(2)
                col1.metric("Likelihood: FAKE", f"{fake_percent:.2f}%")
                col2.metric("Likelihood: REAL", f"{real_percent:.2f}%")
                
                if fake_percent > 50.0:
                    st.error("🚨 **Conclusion:** This article exhibits patterns commonly found in FAKE news.")
                else:
                    st.success("✅ **Conclusion:** This article aligns with patterns commonly found in REAL news.")
                
                with st.expander("View Scraped Text"):
                    st.write(article_text)
                    
st.caption("Note: This is an educational tool. Currently, it runs on a minimal dummy dataset. For high accuracy, you must train the model on a large dataset like the ISOT Fake News corpus.")
