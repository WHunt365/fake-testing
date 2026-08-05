"""
core.py — Fake News Detection pipeline logic.

Shared by app.py (Streamlit UI) and usable standalone from the command line.
Handles: loading the 4 dataset formats, text cleaning, TF-IDF feature
extraction, training the 5 classifiers, and computing evaluation metrics.
"""

import re
import string

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# TEXT PREPROCESSING
# ---------------------------------------------------------------------------

def clean_text(text):
    """Lowercase, strip URLs/HTML/punctuation/digits/extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# DATASET LOADERS
# Each accepts file paths OR file-like objects (e.g. from st.file_uploader)
# and returns a DataFrame with exactly two columns: "text", "label"
# label: 1 = fake, 0 = real
# ---------------------------------------------------------------------------

def load_isot(fake_file, true_file):
    fake = pd.read_csv(fake_file)
    real = pd.read_csv(true_file)
    fake["label"] = 1
    real["label"] = 0
    df = pd.concat([fake, real], ignore_index=True)
    title_col = "title" if "title" in df.columns else None
    text_col = "text" if "text" in df.columns else df.columns[0]
    df["text"] = ((df[title_col].fillna("") + " ") if title_col else "") + df[text_col].fillna("")
    return df[["text", "label"]]


def load_liar(tsv_file):
    cols = ["id", "label", "statement", "subject", "speaker", "job", "state",
            "party", "barely_true_c", "false_c", "half_true_c",
            "mostly_true_c", "pants_fire_c", "context"]
    df = pd.read_csv(tsv_file, sep="\t", header=None, names=cols)
    fake_labels = {"barely-true", "false", "pants-fire"}
    df["label"] = df["label"].apply(lambda x: 1 if x in fake_labels else 0)
    df["text"] = df["statement"]
    return df[["text", "label"]]


def load_kaggle_fake_news(csv_file):
    df = pd.read_csv(csv_file)
    title_col = "title" if "title" in df.columns else None
    text_col = "text" if "text" in df.columns else df.columns[0]
    df["text"] = ((df[title_col].fillna("") + " ") if title_col else "") + df[text_col].fillna("")
    return df[["text", "label"]]


def load_fakenewsnet(fake_files, real_files, text_col="title"):
    fakes = [pd.read_csv(f) for f in fake_files]
    reals = [pd.read_csv(f) for f in real_files]
    fake = pd.concat(fakes, ignore_index=True)
    real = pd.concat(reals, ignore_index=True)
    fake["label"] = 1
    real["label"] = 0
    df = pd.concat([fake, real], ignore_index=True)
    df["text"] = df[text_col].fillna("")
    return df[["text", "label"]]


def load_generic_csv(csv_file, text_col, label_col):
    """For any CSV with a free-text column and a 0/1 (or fake/real) label column."""
    df = pd.read_csv(csv_file)
    df = df.rename(columns={text_col: "text", label_col: "label"})
    if df["label"].dtype == object:
        mapping = {"fake": 1, "real": 0, "FAKE": 1, "REAL": 0, "1": 1, "0": 0}
        df["label"] = df["label"].map(lambda v: mapping.get(str(v).strip(), v))
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]]


# ---------------------------------------------------------------------------
# CLASSIFIERS
# ---------------------------------------------------------------------------

def get_classifiers():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    }


# ---------------------------------------------------------------------------
# TRAIN + EVALUATE
# ---------------------------------------------------------------------------

def run_pipeline(df, max_features=5000, test_size=TEST_SIZE, progress_callback=None):
    """
    df: DataFrame with "text" and "label" columns.
    progress_callback: optional function(fraction: float, message: str),
                        called after each classifier finishes — lets a
                        Streamlit progress bar update live.

    Returns a dict with:
      "metrics_df": DataFrame of accuracy/precision/recall/F1 per algorithm
      "confusion_matrices": {algorithm_name: 2x2 array}
      "vectorizer": fitted TfidfVectorizer
      "models": {algorithm_name: fitted classifier}
      "n_train", "n_test": split sizes
    """
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"]).copy()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0]

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=test_size,
        random_state=RANDOM_STATE, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    classifiers = get_classifiers()
    n_models = len(classifiers)
    results, matrices, fitted_models = [], {}, {}

    for i, (name, clf) in enumerate(classifiers.items(), start=1):
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)

        results.append({
            "Algorithm": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1-score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        })
        matrices[name] = confusion_matrix(y_test, y_pred)
        fitted_models[name] = clf

        if progress_callback:
            progress_callback(i / n_models, f"Trained {name}")

    return {
        "metrics_df": pd.DataFrame(results),
        "confusion_matrices": matrices,
        "vectorizer": vectorizer,
        "models": fitted_models,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "class_balance": df["label"].value_counts().to_dict(),
    }


def predict_single_text(text, vectorizer, model):
    """Classify one custom piece of text using an already-trained model."""
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
    return pred, proba
