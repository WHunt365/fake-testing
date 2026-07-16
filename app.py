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

MODEL_FILE = 'accurate_fake_news_model.joblib'
DATASET_FILE = 'fake_or_real_news.zip'

# --- 1. HELPER FUNCTIONS ---

@st.cache_resource(show_spinner="Training AI on 6,300 articles... This takes about 1 minute on first run!")
def load_or_train_model():
    """Loads the model if it exists, otherwise trains it using the local CSV file."""
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    
    # 1. Check if the dataset exists in the GitHub repo
    if not os.path.exists(DATASET_FILE):
        st.error(f"Dataset missing! Please upload '{DATASET_FILE}' to your GitHub repository.")
        return None

    try:
        # Load the dataset
        df = pd.read_csv(DATASET_FILE)
    except Exception as e:
        st.error(f"Failed to read the dataset. Error: {e}")
        return None

    # Ensure the columns are correct and drop empty rows
    if 'text' not in df.columns or 'label' not in df.columns:
        st.error("The CSV file must have 'text' and 'label' columns.")
        return None
        
    df = df.dropna(subset=['text', 'label'])
    
    # 2. Build the Machine Learning Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # 3. Train the model
    pipeline.fit(df['text'], df['label'])
    
    # 4. Save the model so it never has to train again
    joblib.dump(pipeline, MODEL_FILE)
    
    return pipeline

def scrape_article_text(url):
    """Scrapes paragraph text from a given news article URL."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs]).strip()
        
        return article_text
    except Exception as e:
        st.error(f"Could not scrape URL. The website might have anti-bot protection. Details: {e}")
        return None

# --- 2. STREAMLIT UI ---

st.set_page_config(page_title="Fake News AI Detector", page_icon="🕵️‍♂️")

st.title("🕵️‍♂️ Fake News AI Detector")
st.markdown("""
Paste the URL of a news article below. Our Random Forest AI has been trained on **over 6,300 real and fake news articles** to accurately analyze text patterns, emotional language, and structure to predict authenticity.
""")

model = load_or_train_model()

if model:
    url_input = st.text_input("Paste News Article URL here:", placeholder="https://www.bbc.com/news/...")

    if st.button("Analyze Article", type="primary"):
        if not url_input:
            st.warning("Please enter a URL first!")
        else:
            with st.spinner("Scraping article and analyzing text patterns..."):
                article_text = scrape_article_text(url_input)
                
                if not article_text or len(article_text) < 150:
                    st.error("Could not extract enough text from this URL. The site might be blocking web scrapers, or it's a video-only article.")
                else:
                    st.success(f"Successfully extracted {len(article_text)} characters of text!")
                    
                    classes = model.classes_  
                    probabilities = model.predict_proba([article_text])[0]
                    prob_dict = dict(zip(classes, probabilities))
                    
                    fake_percent = prob_dict.get('FAKE', prob_dict.get('Fake', prob_dict.get('1', 0.0))) * 100
                    real_percent = prob_dict.get('REAL', prob_dict.get('Real', prob_dict.get('0', 0.0))) * 100
                    
                    st.markdown("### Detection Results")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Likelihood: FAKE", f"{fake_percent:.2f}%")
                    col2.metric("Likelihood: REAL", f"{real_percent:.2f}%")
                    
                    if fake_percent > 50.0:
                        st.error("🚨 **Conclusion:** This article exhibits patterns heavily associated with FAKE news.")
                    else:
                        st.success("✅ **Conclusion:** This article aligns with patterns commonly found in REAL news.")
                    
                    with st.expander("View the exact text the AI analyzed"):
                        st.write(article_text)
