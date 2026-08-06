"""
core.py — Fake News Detection pipeline logic.

Shared by app.py (Streamlit UI) and usable standalone from the command line.
Handles: loading the 4 dataset formats, text cleaning, TF-IDF feature
extraction, training the 5 classifiers, and computing evaluation metrics.
"""

import gzip
import re
import string
import zipfile

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
#
# Only WELFake has a schema that's independently documented and confirmed
# (by its authors' Zenodo record and the Kaggle listing): columns
# title, text, label, where label 0 = fake, 1 = real. That loader is
# hardcoded below and automatically flips the label to match this
# project's 1 = fake / 0 = real convention.
#
# BharatFakeNewsKosh, the mahdimashayekhi "Fake News Detection Dataset",
# and the khushikyad001 "Fake News Detection" dataset do not have a
# publicly documented column schema at the time this was written, so
# guessing exact column names here would risk silently mislabeling your
# data. Instead, use `suggest_columns()` to auto-detect likely columns
# after upload, and `load_with_mapping()` with the confirmed column
# names/label value once you've checked a preview of the file (the
# Streamlit app does this for you interactively).
# ---------------------------------------------------------------------------

def read_csv_any(file_obj):
    """Read a CSV from a plain .csv file, or transparently from a .zip
    or .gz compressed file. Accepts either a file path (str) or a
    file-like object such as the one returned by st.file_uploader.

    For .zip archives, the first *.csv member found inside is used.
    """
    is_path = isinstance(file_obj, str)
    name = file_obj if is_path else (getattr(file_obj, "name", "") or "")
    name_lower = str(name).lower()

    if not is_path:
        file_obj.seek(0)

    is_zip = name_lower.endswith(".zip") or (not is_path and zipfile.is_zipfile(file_obj))
    if not is_path:
        file_obj.seek(0)

    if is_zip:
        with zipfile.ZipFile(file_obj) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError("No .csv file found inside the uploaded .zip archive.")
            with zf.open(csv_names[0]) as f:
                df = pd.read_csv(f)
        if not is_path:
            file_obj.seek(0)
        return df

    if name_lower.endswith(".gz") or name_lower.endswith(".gzip"):
        with gzip.open(file_obj, "rt", encoding="utf-8", errors="replace") as f:
            df = pd.read_csv(f)
        if not is_path:
            file_obj.seek(0)
        return df

    df = pd.read_csv(file_obj)
    if not is_path:
        file_obj.seek(0)
    return df


def load_welfake(csv_file):
    """WELFake_Dataset.csv — columns: [index], title, text, label
    (label: 0 = fake, 1 = real in the source data; flipped here).
    Accepts plain .csv, or .zip/.gz containing it."""
    df = read_csv_any(csv_file)
    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    df = df.drop(columns=unnamed_cols, errors="ignore")
    title_col = "title" if "title" in df.columns else None
    text_col = "text" if "text" in df.columns else df.columns[-1]
    df["text"] = ((df[title_col].fillna("") + " ") if title_col else "") + df[text_col].fillna("")
    df["label"] = df["label"].apply(lambda x: 0 if int(x) == 1 else 1)  # flip to 1=fake, 0=real
    return df[["text", "label"]]


def suggest_columns(df):
    """Best-effort guess at text/title/label columns for a dataset whose
    schema isn't hardcoded. Always show this guess to the user for
    confirmation before training — don't trust it silently."""
    text_candidates = ["text", "content", "article", "article_text", "body",
                        "statement", "news", "news_text", "full_text"]
    title_candidates = ["title", "headline", "heading"]
    label_candidates = ["label", "class", "type", "category", "target",
                         "is_fake", "fake_or_real", "outcome", "result"]

    cols_lower = {str(c).lower(): c for c in df.columns}

    def find(cands):
        for c in cands:
            if c in cols_lower:
                return cols_lower[c]
        return None

    return {
        "text_col": find(text_candidates) or (df.columns[0] if len(df.columns) else None),
        "title_col": find(title_candidates),
        "label_col": find(label_candidates),
    }


def load_with_mapping(csv_file, text_col, label_col, fake_value, title_col=None):
    """
    Generic loader with an explicit column mapping the user confirms.
    Used for BharatFakeNewsKosh, the mahdimashayekhi and khushikyad001
    datasets, and any other custom CSV.

    fake_value: the raw value in label_col that represents "fake"
                (e.g. 1, "FAKE", "fake") — everything else is treated
                as real. Compared as a lowercased string, so it works
                for both numeric and text labels.

    Accepts plain .csv, or .zip/.gz containing it.
    """
    df = read_csv_any(csv_file)
    text_series = df[text_col].fillna("")
    if title_col and title_col in df.columns and title_col != text_col:
        text_series = df[title_col].fillna("") + " " + text_series
    df["text"] = text_series
    df["label"] = df[label_col].apply(
        lambda v: 1 if str(v).strip().lower() == str(fake_value).strip().lower() else 0
    )
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
