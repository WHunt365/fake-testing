import streamlit as st
import re
import string
import warnings
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# 1. TEXT PREPROCESSING[cite: 2]
# ---------------------------------------------------------------------------
def clean_text(text):
    """Lowercase, strip URLs/punctuation/digits/extra whitespace[cite: 2]."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # remove URLs[cite: 2]
    text = re.sub(r"<.*?>", " ", text)                       # remove HTML tags[cite: 2]
    text = text.translate(str.maketrans("", "", string.punctuation)) # remove punctuation[cite: 2]
    text = re.sub(r"\d+", " ", text)                          # remove digits[cite: 2]
    text = re.sub(r"\s+", " ", text).strip()                  # collapse whitespace[cite: 2]
    return text

# ---------------------------------------------------------------------------
# 2. DATASET LOADERS[cite: 2]
# ---------------------------------------------------------------------------
@st.cache_data
def load_isot(fake_file, true_file):
    fake = pd.read_csv(fake_file)
    real = pd.read_csv(true_file)
    fake["label"] = 1
    real["label"] = 0
    df = pd.concat([fake, real], ignore_index=True)
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna(""))
    return df[["text", "label"]]

@st.cache_data
def load_liar(train_tsv_file):
    cols = ["id", "label", "statement", "subject", "speaker", "job", "state",
            "party", "barely_true_c", "false_c", "half_true_c",
            "mostly_true_c", "pants_fire_c", "context"]
    df = pd.read_csv(train_tsv_file, sep="\t", header=None, names=cols)

    # Collapse LIAR's 6-way truthfulness scale into a binary fake/real label[cite: 2]
    fake_labels = {"barely-true", "false", "pants-fire"}
    df["label"] = df["label"].apply(lambda x: 1 if x in fake_labels else 0)
    df["text"] = df["statement"]
    return df[["text", "label"]]

@st.cache_data
def load_kaggle_fake_news(train_csv_file):
    df = pd.read_csv(train_csv_file)
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna(""))
    return df[["text", "label"]]

@st.cache_data
def load_fakenewsnet(politifact_fake, politifact_real, gossipcop_fake, gossipcop_real, text_col="title"):
    fakes = [pd.read_csv(politifact_fake), pd.read_csv(gossipcop_fake)]
    reals = [pd.read_csv(politifact_real), pd.read_csv(gossipcop_real)]
    fake = pd.concat(fakes, ignore_index=True)
    real = pd.concat(reals, ignore_index=True)
    fake["label"] = 1
    real["label"] = 0
    df = pd.concat([fake, real], ignore_index=True)
    df["text"] = df[text_col].fillna("")
    return df[["text", "label"]]

# ---------------------------------------------------------------------------
# 3. TRAIN + EVALUATE CLASSIFIERS[cite: 2]
# ---------------------------------------------------------------------------
def get_classifiers():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="linear", random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    }

@st.cache_data
def evaluate_dataset(df, dataset_name, max_features=5000):
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0]

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    results = []
    for name, clf in get_classifiers().items():
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)

        results.append({
            "Dataset": dataset_name,
            "Algorithm": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1-score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        })

    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# 4. STREAMLIT UI Build
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Fake News Classifiers", layout="wide")
st.title("Fake News Detection: Comparative Analysis[cite: 2]")
st.markdown("Upload your datasets to train and evaluate 5 classification algorithms independently[cite: 2].")

# Sidebar Uploaders
st.sidebar.header("1. ISOT Dataset")
isot_fake = st.sidebar.file_uploader("ISOT Fake.csv", type=["csv"])
isot_true = st.sidebar.file_uploader("ISOT True.csv", type=["csv"])

st.sidebar.header("2. LIAR Dataset")
liar_train = st.sidebar.file_uploader("LIAR train.tsv", type=["tsv"])

st.sidebar.header("3. Kaggle Fake News")
kaggle_train = st.sidebar.file_uploader("Kaggle train.csv", type=["csv"])

st.sidebar.header("4. FakeNewsNet")
p_fake = st.sidebar.file_uploader("politifact_fake.csv", type=["csv"])
p_real = st.sidebar.file_uploader("politifact_real.csv", type=["csv"])
g_fake = st.sidebar.file_uploader("gossipcop_fake.csv", type=["csv"])
g_real = st.sidebar.file_uploader("gossipcop_real.csv", type=["csv"])

if st.button("Run Evaluation", type="primary"):
    all_results = []

    # Run ISOT
    if isot_fake and isot_true:
        with st.spinner("Processing ISOT Dataset..."):
            df_isot = load_isot(isot_fake, isot_true)
            res_isot = evaluate_dataset(df_isot, "ISOT Fake News Dataset")
            all_results.append(res_isot)
            st.subheader("ISOT Fake News Dataset Results")
            st.dataframe(res_isot, hide_index=True, use_container_width=True)
            
    # Run LIAR
    if liar_train:
        with st.spinner("Processing LIAR Dataset..."):
            df_liar = load_liar(liar_train)
            res_liar = evaluate_dataset(df_liar, "LIAR Dataset")
            all_results.append(res_liar)
            st.subheader("LIAR Dataset Results")
            st.dataframe(res_liar, hide_index=True, use_container_width=True)

    # Run Kaggle
    if kaggle_train:
        with st.spinner("Processing Kaggle Fake News Dataset..."):
            df_kaggle = load_kaggle_fake_news(kaggle_train)
            res_kaggle = evaluate_dataset(df_kaggle, "Kaggle Fake News Dataset")
            all_results.append(res_kaggle)
            st.subheader("Kaggle Dataset Results")
            st.dataframe(res_kaggle, hide_index=True, use_container_width=True)

    # Run FakeNewsNet
    if p_fake and p_real and g_fake and g_real:
        with st.spinner("Processing FakeNewsNet Dataset..."):
            df_fnn = load_fakenewsnet(p_fake, p_real, g_fake, g_real, text_col="title")
            res_fnn = evaluate_dataset(df_fnn, "FakeNewsNet")
            all_results.append(res_fnn)
            st.subheader("FakeNewsNet Dataset Results")
            st.dataframe(res_fnn, hide_index=True, use_container_width=True)

    # Final Summary Table
    if not all_results:
        st.warning("Please upload the required files for at least one dataset in the sidebar to run the evaluation.")
    else:
        st.success("Evaluation Complete!")
        st.divider()
        st.subheader("Summary: Cross-Dataset Average per Algorithm")
        
        final = pd.concat(all_results, ignore_index=True)
        summary = final.groupby("Algorithm")[["Accuracy", "Precision", "Recall", "F1-score"]].mean().round(4).reset_index()
        
        # Maintain manuscript order
        sorter = ['Logistic Regression', 'Random Forest', 'SVM', 'K-Nearest Neighbors', 'Decision Tree']
        summary['Algorithm'] = pd.Categorical(summary['Algorithm'], categories=sorter, ordered=True)
        summary = summary.sort_values('Algorithm')
        
        st.dataframe(summary, hide_index=True, use_container_width=True)
