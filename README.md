# Fake News Detection — Classifier Comparison App

An interactive Streamlit app for the group project. Upload any of the
four project datasets (or your own CSV), and it trains and compares five
classifiers — Logistic Regression, Random Forest, SVM, K-Nearest
Neighbors, and Decision Tree — showing accuracy, precision, recall,
F1-score, confusion matrices, and a box to test your own headline.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit UI |
| `core.py` | Data loading, preprocessing, TF-IDF, training/evaluation logic (no Streamlit code — reusable and independently testable) |
| `.streamlit/config.toml` | Raises the file-upload limit to 500MB (Streamlit's default is 200MB) |
| `requirements.txt` | Python dependencies |

## 1. Run it locally

```bash
# 1. Clone your repo (see Part 2 below) or cd into this folder
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

This opens the app at `http://localhost:8501`. Use the sidebar to pick a
dataset, upload the corresponding file(s), then click
**"Run all 5 classifiers."**

Datasets used in this project:
- **WELFake Dataset** — schema is confirmed (title, text, label), so this
  one just works after upload: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
- **BharatFakeNewsKosh** — https://www.kaggle.com/datasets/man2191989/bharatfakenewskosh
- **Fake News Detection Dataset** (mahdimashayekhi) — https://www.kaggle.com/datasets/mahdimashayekhi/fake-news-detection-dataset
- **Fake News Detection** (khushikyad001) — https://www.kaggle.com/datasets/khushikyad001/fake-news-detection

The last three don't have a publicly documented column schema, so after
you upload one the app shows a preview, guesses the likely text/title/label
columns, and asks you to confirm (or correct) the mapping — including
which value in the label column means "fake" — before training. This
avoids silently mislabeling data based on a guessed schema.

### Uploading large files (e.g. WELFake_Dataset.csv, ~245MB)

Streamlit's file uploader defaults to a 200MB limit, which is too small
for WELFake and would reject the upload. This repo includes
`.streamlit/config.toml` with `maxUploadSize = 500`, which raises the
limit to 500MB. As long as you keep that file in place (it's committed
to the repo, not gitignored), the limit applies automatically both
locally and on Streamlit Community Cloud — no extra flags needed.

If you ever need a different limit, edit the number in
`.streamlit/config.toml` directly, or override it for a single local run
without editing the file:

```bash
streamlit run app.py --server.maxUploadSize 500
```

Note: Streamlit Community Cloud's own infrastructure has historically
enforced its own hard ceiling on request size regardless of this
setting. If a very large file still fails to upload after deploying,
run the app locally instead (`streamlit run app.py`), or pre-filter /
compress the dataset before upload.

## 2. Push this project to GitHub

```bash
cd streamlit_app          # this folder
git init
git add .
git commit -m "Initial commit: fake news classifier comparison app"

# Create an empty repo on github.com first (no README/license), then:
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

If you're working as a group, add your teammates as collaborators on
GitHub (Settings → Collaborators), or have each member fork and open
pull requests.

## 3. Deploy for free on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with your GitHub account.
2. Click **"New app."**
3. Select your repository, branch (`main`), and set **Main file path**
   to `app.py`.
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt`
   automatically and gives you a public URL like
   `https://<your-app-name>.streamlit.app` you can put in your paper or
   share with your professor.
5. Any time you `git push` new changes, the deployed app updates
   automatically within a minute or two.

## Notes

- The datasets themselves are **not** included in this repo (see
  `.gitignore`) since they're large and publicly downloadable — users
  upload their own copy through the app's file uploader.
- `core.py` has no Streamlit-specific code, so your group can also run
  it directly from the command line or a Jupyter notebook if you'd
  rather generate the paper's result tables that way instead of via
  the UI.
- The "Generic CSV" option in the sidebar lets you test any dataset
  that isn't one of the four main ones, as long as it has one text
  column and one label column.
