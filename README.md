# FERC Data Explorer (Streamlit)

This repository contains a minimal Streamlit dashboard (`app.py`) that reads the SQLite database `ferc-eqr-scraper/ferc_data.db` and exposes tables, CSV download, and a quick Altair chart.

Quick start (local):

```bash
# from repository root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Quick smoke-test (without running the server):

```bash
# from repository root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import streamlit as st, pandas as pd, altair as alt, sqlite3; print('imports OK')"
```

Deploy to Streamlit Community Cloud (fastest):

1. Ensure `app.py`, `requirements.txt` and the DB `ferc-eqr-scraper/ferc_data.db` are committed to your GitHub repo. Note: GitHub file size limit is 100 MB per file. If the DB is larger, either use Git LFS or host the DB externally (S3, GCS) and modify `app.py` to download it at startup.
2. Go to https://share.streamlit.io/ and create a new app. Connect your GitHub repo, pick the branch, and set `app.py` as the entrypoint. Deploy.

If you prefer Hugging Face Spaces (Streamlit runtime): create a new Space, upload the files (`app.py`, `requirements.txt`, DB) or push via `git`, and the Space will provide a public URL.

Notes and next steps:
- This is a simple read-only viewer. For heavy traffic, concurrent users, or write access, migrate to a server-backed DB (Postgres, Supabase) and update the app to use a proper DB connection.
- The app serves the DB contents publicly when deployed—do not include PII or sensitive data in a public deployment.
- I can prepare a version that downloads the DB from S3 at startup if your DB is too large for GitHub.

