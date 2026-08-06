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
    load_welfake, suggest_columns, load_with_mapping,
    run_pipeline, predict_single_text,
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
#
# WELFake has a schema confirmed by its authors (title, text, label),
# hardcoded in core.load_welfake(). The other three datasets below don't
# have a publicly documented schema, so this app shows you a preview and
# auto-detected column guesses and asks you to confirm them before
# training — rather than assuming column names that might be wrong.
# ---------------------------------------------------------------------------

st.sidebar.header("1. Choose a dataset")
dataset_choice = st.sidebar.selectbox(
    "Dataset",
    ["WELFake Dataset", "BharatFakeNewsKosh",
     "Fake News Detection Dataset (mahdimashayekhi)",
     "Fake News Detection (khushikyad001)",
     "Generic CSV (any other file)"],
)

df = None

if dataset_choice == "WELFake Dataset":
    st.sidebar.markdown(
        "Upload `WELFake_Dataset.csv`. Schema (title, text, label) is "
        "confirmed from the dataset's documentation — no setup needed."
    )
    welfake_f = st.sidebar.file_uploader("WELFake_Dataset.csv", type="csv", key="welfake")
    if welfake_f:
        df = load_welfake(welfake_f)

else:  # BharatFakeNewsKosh, mahdimashayekhi, khushikyad001, or Generic CSV
    st.sidebar.markdown(
        "This dataset's column names aren't hardcoded — upload it, "
        "then confirm which columns to use below."
    )
    upload_key = dataset_choice.replace(" ", "_")
    uploaded_f = st.sidebar.file_uploader("CSV file", type="csv", key=upload_key)

    if uploaded_f:
        preview_df = pd.read_csv(uploaded_f)
        uploaded_f.seek(0)
        guess = suggest_columns(preview_df)
        columns = list(preview_df.columns)

        def idx_or_0(col):
            return columns.index(col) if col in columns else 0

        st.sidebar.caption(f"Detected {len(preview_df):,} rows, {len(columns)} columns.")
        text_col = st.sidebar.selectbox(
            "Text / article column", columns, index=idx_or_0(guess["text_col"]), key=f"{upload_key}_text"
        )
        title_options = ["(none)"] + columns
        title_default = title_options.index(guess["title_col"]) if guess["title_col"] in title_options else 0
        title_col_choice = st.sidebar.selectbox(
            "Title column (optional, combined with text)", title_options,
            index=title_default, key=f"{upload_key}_title"
        )
        title_col = None if title_col_choice == "(none)" else title_col_choice
        label_col = st.sidebar.selectbox(
            "Label column", columns, index=idx_or_0(guess["label_col"]), key=f"{upload_key}_label"
        )

        unique_vals = preview_df[label_col].dropna().unique().tolist()[:20]
        fake_value = st.sidebar.selectbox(
            "Which value means FAKE?", unique_vals, key=f"{upload_key}_fakeval"
        )

        df = load_with_mapping(uploaded_f, text_col=text_col, label_col=label_col,
                                fake_value=fake_value, title_col=title_col)

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
