import streamlit as st
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure NLTK resources are downloaded
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

download_nltk_data()

# ==========================================
# 1. Data Preprocessing (Section 2.2)
# ==========================================
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(clean_tokens)

@st.cache_data
def load_and_preprocess(file, text_col, label_col):
    df = pd.read_csv(file)
    df = df.dropna(subset=[text_col, label_col])
    df = df.drop_duplicates(subset=[text_col])
    
    # Sample down for Streamlit performance if dataset is massive
    if len(df) > 10000:
        df = df.sample(10000, random_state=42)
        
    df['clean_text'] = df[text_col].apply(preprocess_text)
    return df['clean_text'], df[label_col]

# ==========================================
# 2. Experimental Setup & Modeling (Sections 2.3 - 2.6)
# ==========================================
@st.cache_data
def evaluate_models_on_dataset(X, y, dataset_name):
    # TF-IDF Feature Extraction
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    # Five distinct standalone algorithms[cite: 1]
    models = {
        'Logistic Regr.': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'SVM': LinearSVC(random_state=42),
        'K-NN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    results = []
    for model_name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Fit and Predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics[cite: 1]
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
        rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        results.append({
            'Algorithm': model_name,
            'Acc.': round(acc, 4),
            'Prec.': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4)
        })
        
    return pd.DataFrame(results)

# ==========================================
# 3. Streamlit UI Build
# ==========================================
st.set_page_config(page_title="Fake News Detection Research", layout="wide")

st.title("Cross-Domain Fake News Detection Experiment")
st.write("""
This application replicates the experimental methodology of the research paper: 
**A Comparative Analysis of Machine Learning Classification Algorithms for Cross-Domain Fake News Detection**[cite: 1].
Upload the four required datasets below to generate the study's findings tables.
""")

st.sidebar.header("Upload Datasets")
st.sidebar.markdown("Please upload the specific CSV files for the four domains[cite: 1]. Ensure they have a `text` (or `statement`) column and a binary `label` column.")

# File uploaders for the four datasets[cite: 1]
isot_file = st.sidebar.file_uploader("1. ISOT Dataset", type=["csv"])
liar_file = st.sidebar.file_uploader("2. LIAR Dataset", type=["csv"])
kaggle_file = st.sidebar.file_uploader("3. Kaggle Fake News", type=["csv"])
fakenewsnet_file = st.sidebar.file_uploader("4. FakeNewsNet", type=["csv"])

datasets_config = {
    'ISOT Fake News Dataset': {'file': isot_file, 'text_col': 'text', 'label_col': 'label'},
    'LIAR Dataset': {'file': liar_file, 'text_col': 'statement', 'label_col': 'label'},
    'Kaggle Fake News Dataset': {'file': kaggle_file, 'text_col': 'text', 'label_col': 'label'},
    'FakeNewsNet': {'file': fakenewsnet_file, 'text_col': 'text', 'label_col': 'label'}
}

if st.button("Run Experiments", type="primary"):
    all_results = {}
    
    for ds_name, config in datasets_config.items():
        if config['file'] is not None:
            with st.spinner(f"Processing and training on {ds_name}..."):
                try:
                    X, y = load_and_preprocess(config['file'], config['text_col'], config['label_col'])
                    df_results = evaluate_models_on_dataset(X, y, ds_name)
                    all_results[ds_name] = df_results
                    
                    st.subheader(f"Performance on the {ds_name}")
                    st.dataframe(df_results, hide_index=True, use_container_width=True)
                except Exception as e:
                    st.error(f"Error processing {ds_name}: Check if the column names match ({config['text_col']}, {config['label_col']}). Error details: {e}")
        else:
            st.warning(f"Awaiting upload for {ds_name}")

    # Generate Table 5: Average Performance
    if len(all_results) > 0:
        st.divider()
        st.subheader("Table 5. Average Performance Across All Evaluated Datasets")
        
        combined_df = pd.concat(all_results.values())
        table_5 = combined_df.groupby('Algorithm').mean().reset_index()
        table_5[['Acc.', 'Prec.', 'Recall', 'F1']] = table_5[['Acc.', 'Prec.', 'Recall', 'F1']].round(4)
        
        # Maintain manuscript order[cite: 1]
        sorter = ['Logistic Regr.', 'Random Forest', 'SVM', 'K-NN', 'Decision Tree']
        table_5['Algorithm'] = pd.Categorical(table_5['Algorithm'], categories=sorter, ordered=True)
        table_5 = table_5.sort_values('Algorithm')
        
        st.dataframe(table_5, hide_index=True, use_container_width=True)
        
        st.success("Experiments completed! You can now copy these metrics directly into your research paper draft.")
