"""
Streamlit app for the Fake News Detection group project.

Lets the user upload one of the 4 project datasets (or any generic
text/label CSV), trains all 5 classifiers (Logistic Regression, Random
Forest, SVM, K-Nearest Neighbors, Decision Tree), and displays accuracy,
precision, recall, and F1-score for each — plus confusion matrices and
a live "test your own headline" box.

Run locally:   streamlit run app.py
Deploy:        push this folder to GitHub, then deploy on
               https://share.streamlit.io pointing at app.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from core import (
    load_isot, load_liar, load_kaggle_fake_news, load_fakenewsnet,
    load_generic_csv, run_pipeline, predict_single_text,
)

st.set_page_config(page_title="Fake News Classifier Comparison", layout="wide")

st.title("📰 Fake News Detection — Classifier Comparison")
st.caption(
    "Compares Logistic Regression, Random Forest, SVM, K-Nearest Neighbors, "
    "and Decision Tree (trained independently, no ensembling) on your chosen dataset."
)

if "results" not in st.session_state:
    st.session_state.results = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

# ---------------------------------------------------------------------------
# SIDEBAR: dataset selection + upload
# ---------------------------------------------------------------------------

st.sidebar.header("1. Choose a dataset")
dataset_choice = st.sidebar.selectbox(
    "Dataset format",
    ["ISOT Fake News Dataset", "LIAR Dataset", "Kaggle Fake News Dataset",
     "FakeNewsNet", "Generic CSV (text + label columns)"],
)

df = None

if dataset_choice == "ISOT Fake News Dataset":
    st.sidebar.markdown("Upload the two ISOT files.")
    fake_f = st.sidebar.file_uploader("Fake.csv", type="csv", key="isot_fake")
    true_f = st.sidebar.file_uploader("True.csv", type="csv", key="isot_true")
    if fake_f and true_f:
        df = load_isot(fake_f, true_f)

elif dataset_choice == "LIAR Dataset":
    st.sidebar.markdown("Upload the LIAR `train.tsv` file (tab-separated, no header).")
    liar_f = st.sidebar.file_uploader("train.tsv", type=["tsv", "csv"], key="liar")
    if liar_f:
        df = load_liar(liar_f)

elif dataset_choice == "Kaggle Fake News Dataset":
    st.sidebar.markdown("Upload the Kaggle competition `train.csv` (columns: title, text, label).")
    kaggle_f = st.sidebar.file_uploader("train.csv", type="csv", key="kaggle")
    if kaggle_f:
        df = load_kaggle_fake_news(kaggle_f)

elif dataset_choice == "FakeNewsNet":
    st.sidebar.markdown("Upload the 4 FakeNewsNet CSVs.")
    pf = st.sidebar.file_uploader("politifact_fake.csv", type="csv", key="pf")
    pr = st.sidebar.file_uploader("politifact_real.csv", type="csv", key="pr")
    gf = st.sidebar.file_uploader("gossipcop_fake.csv", type="csv", key="gf")
    gr = st.sidebar.file_uploader("gossipcop_real.csv", type="csv", key="gr")
    text_col = st.sidebar.text_input("Text column to use", value="title")
    if pf and pr and gf and gr:
        df = load_fakenewsnet([pf, gf], [pr, gr], text_col=text_col)

else:  # Generic CSV
    st.sidebar.markdown("Upload any CSV with a text column and a label column.")
    generic_f = st.sidebar.file_uploader("dataset.csv", type="csv", key="generic")
    if generic_f:
        preview = pd.read_csv(generic_f)
        generic_f.seek(0)
        text_col = st.sidebar.selectbox("Text column", preview.columns, key="generic_text_col")
        label_col = st.sidebar.selectbox("Label column", preview.columns, key="generic_label_col")
        df = load_generic_csv(generic_f, text_col, label_col)

st.sidebar.header("2. Settings")
max_features = st.sidebar.slider("Max TF-IDF features", 500, 20000, 5000, step=500)
test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, step=0.05)

run_clicked = st.sidebar.button("🚀 Run all 5 classifiers", type="primary", disabled=df is None)

# ---------------------------------------------------------------------------
# MAIN: preview + run + results
# ---------------------------------------------------------------------------

if df is not None:
    st.subheader("Dataset preview")
    c1, c2 = st.columns([3, 1])
    with c1:
        st.dataframe(df.head(10), use_container_width=True)
    with c2:
        st.metric("Total rows", len(df))
        st.metric("Fake (1)", int((df["label"] == 1).sum()))
        st.metric("Real (0)", int((df["label"] == 0).sum()))
else:
    st.info("⬅️ Upload dataset file(s) in the sidebar to get started.")

if run_clicked and df is not None:
    progress_bar = st.progress(0.0, text="Starting...")

    def update_progress(fraction, message):
        progress_bar.progress(fraction, text=message)

    with st.spinner("Cleaning text and extracting TF-IDF features..."):
        results = run_pipeline(
            df, max_features=max_features, test_size=test_size,
            progress_callback=update_progress,
        )

    progress_bar.empty()
    st.session_state.results = results
    st.session_state.dataset_name = dataset_choice
    st.success(f"Done — trained on {results['n_train']} articles, tested on {results['n_test']}.")

# ---------------------------------------------------------------------------
# RESULTS DISPLAY (persists across reruns via session_state)
# ---------------------------------------------------------------------------

results = st.session_state.results

if results is not None:
    st.divider()
    st.subheader(f"Results — {st.session_state.dataset_name}")

    metrics_df = results["metrics_df"]
    st.dataframe(
        metrics_df.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1-score"], color="#c6f6d5"
        ),
        use_container_width=True,
    )

    csv_bytes = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download results as CSV", csv_bytes, "results.csv", "text/csv")

    # --- Bar chart comparison ---
    st.subheader("Metric comparison across algorithms")
    fig, ax = plt.subplots(figsize=(9, 4))
    metrics_df.set_index("Algorithm")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
        kind="bar", ax=ax
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig)

    # --- Confusion matrices ---
    st.subheader("Confusion matrices")
    cols = st.columns(5)
    for col, (name, cm) in zip(cols, results["confusion_matrices"].items()):
        with col:
            st.caption(name)
            fig_cm, ax_cm = plt.subplots(figsize=(2.6, 2.4))
            ax_cm.imshow(cm, cmap="Blues")
            for (i, j), val in __import__("numpy").ndenumerate(cm):
                ax_cm.text(j, i, str(val), ha="center", va="center")
            ax_cm.set_xticks([0, 1]); ax_cm.set_xticklabels(["Real", "Fake"], fontsize=8)
            ax_cm.set_yticks([0, 1]); ax_cm.set_yticklabels(["Real", "Fake"], fontsize=8)
            ax_cm.set_xlabel("Predicted", fontsize=8)
            ax_cm.set_ylabel("Actual", fontsize=8)
            st.pyplot(fig_cm)

    # --- Try your own headline ---
    st.divider()
    st.subheader("🔎 Test a custom headline / article")
    algo_choice = st.selectbox("Model to use", list(results["models"].keys()))
    custom_text = st.text_area("Paste a headline or short article here:", height=100)

    if st.button("Classify this text"):
        if custom_text.strip():
            pred, proba = predict_single_text(
                custom_text, results["vectorizer"], results["models"][algo_choice]
            )
            label = "🟥 FAKE" if pred == 1 else "🟩 REAL"
            st.markdown(f"### Prediction: {label}")
            if proba is not None:
                st.write(f"Confidence — Real: {proba[0]:.1%}, Fake: {proba[1]:.1%}")
        else:
            st.warning("Please enter some text first.")
