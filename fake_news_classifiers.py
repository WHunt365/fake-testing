"""
Fake News Detection: Comparative Analysis of 5 Classification Algorithms
==========================================================================

Pipeline: load dataset -> preprocess text -> TF-IDF features -> train/test
split -> train 5 classifiers independently -> evaluate with accuracy,
precision, recall, F1-score -> save results to CSV.

Algorithms compared: Logistic Regression, Random Forest, SVM,
K-Nearest Neighbors, Decision Tree.

HOW TO USE
----------
1. pip install pandas scikit-learn
2. Download the 4 datasets below into data/<name>/
3. For the 3 datasets marked "schema not fixed" below, open the CSV once
   in Excel/pandas, note the actual column names, and fill in the
   matching CONFIG entry near the top of main(). This script deliberately
   does NOT guess those column names for you — guessing wrong would
   silently mislabel your data.
4. Run: python fake_news_classifiers.py

DATASET DOWNLOAD LINKS
-----------------------
1. WELFake Dataset (schema confirmed — no setup needed):
   https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
   -> WELFake_Dataset.csv, columns: [index], title, text, label
      (label: 0 = fake, 1 = real in the source file — this script flips
      it to this project's convention of 1 = fake, 0 = real)

2. BharatFakeNewsKosh (schema not fixed — confirm columns yourself):
   https://www.kaggle.com/datasets/man2191989/bharatfakenewskosh

3. Fake News Detection Dataset, by mahdimashayekhi (schema not fixed):
   https://www.kaggle.com/datasets/mahdimashayekhi/fake-news-detection-dataset

4. Fake News Detection, by khushikyad001 (schema not fixed):
   https://www.kaggle.com/datasets/khushikyad001/fake-news-detection

For datasets 2-4, this project's Streamlit app (see streamlit_app/) is the
easier option: it shows you a live preview, auto-guesses likely text/label
columns, and lets you confirm the mapping interactively before training —
worth using instead of this CLI script if you're not sure about a schema.
"""

import os
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
# 1. TEXT PREPROCESSING
# ---------------------------------------------------------------------------

def clean_text(text):
    """Lowercase, strip URLs/punctuation/digits/extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # remove URLs
    text = re.sub(r"<.*?>", " ", text)                       # remove HTML tags
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)                          # remove digits
    text = re.sub(r"\s+", " ", text).strip()                  # collapse whitespace
    return text


# ---------------------------------------------------------------------------
# 2. DATASET LOADERS
#    Each loader returns a DataFrame with exactly two columns: "text", "label"
#    label: 1 = fake, 0 = real
# ---------------------------------------------------------------------------

def load_welfake(csv_path):
    """WELFake_Dataset.csv — columns: [index], title, text, label
    (label: 0 = fake, 1 = real in the source data; flipped here to this
    project's 1 = fake / 0 = real convention)."""
    df = pd.read_csv(csv_path)
    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    df = df.drop(columns=unnamed_cols, errors="ignore")
    title_col = "title" if "title" in df.columns else None
    text_col = "text" if "text" in df.columns else df.columns[-1]
    df["text"] = ((df[title_col].fillna("") + " ") if title_col else "") + df[text_col].fillna("")
    df["label"] = df["label"].apply(lambda x: 0 if int(x) == 1 else 1)
    return df[["text", "label"]]


def load_with_mapping(csv_path, text_col, label_col, fake_value, title_col=None):
    """
    Generic loader for a dataset whose column names you've confirmed
    yourself (see CONFIG in main() below).

    fake_value: the raw value in label_col that means "fake" (e.g. 1,
    "FAKE", "fake") — everything else is treated as real. Compared as a
    lowercased string, so it works whether your label column is numeric
    or text.
    """
    df = pd.read_csv(csv_path)
    text_series = df[text_col].fillna("")
    if title_col and title_col in df.columns and title_col != text_col:
        text_series = df[title_col].fillna("") + " " + text_series
    df["text"] = text_series
    df["label"] = df[label_col].apply(
        lambda v: 1 if str(v).strip().lower() == str(fake_value).strip().lower() else 0
    )
    return df[["text", "label"]]


# ---------------------------------------------------------------------------
# 3. TRAIN + EVALUATE 5 CLASSIFIERS (standalone, no ensembling between them)
# ---------------------------------------------------------------------------

def get_classifiers():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="linear", random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    }


def evaluate_dataset(df, dataset_name, max_features=5000):
    """Clean text, vectorize with TF-IDF, train+test all 5 classifiers.
    Returns a DataFrame of accuracy/precision/recall/F1 per algorithm."""

    print(f"\n{'=' * 60}\nDataset: {dataset_name}  (n = {len(df)})\n{'=' * 60}")

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
        print(f"  {name:<22} done")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 4. MAIN — point the paths below at your downloaded dataset files.
#    For the 3 datasets with no fixed schema, fill in the matching CONFIG
#    entry once you've looked at the CSV's actual column names.
# ---------------------------------------------------------------------------

# Fill these in after inspecting each CSV's header row yourself:
#   text_col   -> the column holding the article/headline text
#   label_col  -> the column holding the fake/real label
#   fake_value -> the exact value in label_col that means "fake"
#                 (e.g. 1, "FAKE", "fake" — check with df[label_col].unique())
#   title_col  -> optional second text column to prepend (e.g. a headline)
CONFIG = {
    "bharat": {"text_col": None, "label_col": None, "fake_value": None, "title_col": None},
    "mahdimashayekhi": {"text_col": None, "label_col": None, "fake_value": None, "title_col": None},
    "khushikyad001": {"text_col": None, "label_col": None, "fake_value": None, "title_col": None},
}


def main():
    all_results = []

    # --- 1. WELFake ---
    if os.path.exists("data/welfake/WELFake_Dataset.csv"):
        df_welfake = load_welfake("data/welfake/WELFake_Dataset.csv")
        all_results.append(evaluate_dataset(df_welfake, "WELFake Dataset"))
    else:
        print("Skipping WELFake: place WELFake_Dataset.csv in data/welfake/")

    # --- 2. BharatFakeNewsKosh ---
    cfg = CONFIG["bharat"]
    path = "data/bharat/bharatfakenewskosh.csv"
    if os.path.exists(path) and cfg["text_col"] and cfg["label_col"] and cfg["fake_value"] is not None:
        df_bharat = load_with_mapping(path, **cfg)
        all_results.append(evaluate_dataset(df_bharat, "BharatFakeNewsKosh"))
    elif os.path.exists(path):
        print("Skipping BharatFakeNewsKosh: fill in CONFIG['bharat'] with the real column names first.")
    else:
        print("Skipping BharatFakeNewsKosh: place the csv in data/bharat/")

    # --- 3. mahdimashayekhi Fake News Detection Dataset ---
    cfg = CONFIG["mahdimashayekhi"]
    path = "data/mahdimashayekhi/fake_news_detection_dataset.csv"
    if os.path.exists(path) and cfg["text_col"] and cfg["label_col"] and cfg["fake_value"] is not None:
        df_mahdi = load_with_mapping(path, **cfg)
        all_results.append(evaluate_dataset(df_mahdi, "Fake News Detection Dataset (mahdimashayekhi)"))
    elif os.path.exists(path):
        print("Skipping mahdimashayekhi dataset: fill in CONFIG['mahdimashayekhi'] with the real column names first.")
    else:
        print("Skipping mahdimashayekhi dataset: place the csv in data/mahdimashayekhi/")

    # --- 4. khushikyad001 Fake News Detection ---
    cfg = CONFIG["khushikyad001"]
    path = "data/khushikyad001/fake_news_detection.csv"
    if os.path.exists(path) and cfg["text_col"] and cfg["label_col"] and cfg["fake_value"] is not None:
        df_khushi = load_with_mapping(path, **cfg)
        all_results.append(evaluate_dataset(df_khushi, "Fake News Detection (khushikyad001)"))
    elif os.path.exists(path):
        print("Skipping khushikyad001 dataset: fill in CONFIG['khushikyad001'] with the real column names first.")
    else:
        print("Skipping khushikyad001 dataset: place the csv in data/khushikyad001/")

    if not all_results:
        print("\nNo datasets ran. Check the messages above, add files, and/or fill in CONFIG.")
        return

    final = pd.concat(all_results, ignore_index=True)
    final.to_csv("results_per_dataset.csv", index=False)
    print("\nSaved per-dataset results to results_per_dataset.csv")

    # Cross-dataset average per algorithm (for the paper's summary table)
    summary = final.groupby("Algorithm")[["Accuracy", "Precision", "Recall", "F1-score"]].mean().round(4)
    summary = summary.reset_index()
    summary.to_csv("results_summary_avg.csv", index=False)
    print("Saved cross-dataset averages to results_summary_avg.csv\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
