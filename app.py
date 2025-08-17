import streamlit as st
import pandas as pd
import sqlite3
import altair as alt
import os
from typing import List


st.set_page_config(page_title="FERC Data Explorer", layout="wide")

# Path to the SQLite DB inside this repo. Adjust if you place it elsewhere.
DEFAULT_DB_PATH = os.path.join("ferc-eqr-scraper", "ferc_data.db")


@st.cache_resource
def get_conn(db_path: str):
    """Return a cached sqlite3.Connection resource. Raises FileNotFoundError if missing."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")
    return sqlite3.connect(db_path, check_same_thread=False)


@st.cache_data(ttl=600)
def list_tables(db_path: str) -> List[str]:
    """List table names. Accepts db_path (serializable) so Streamlit can cache the result."""
    q = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    with sqlite3.connect(db_path) as conn:
        return [r[0] for r in conn.execute(q).fetchall()]


@st.cache_data(ttl=600)
def read_table(db_path: str, table_name: str, limit: int = 1000) -> pd.DataFrame:
    q = f"SELECT * FROM \"{table_name}\" LIMIT {int(limit)}"
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(q, conn)


def main():
    st.title("FERC Data Explorer")

    st.sidebar.header("Configuration")
    db_path = st.sidebar.text_input("Path to SQLite DB", value=DEFAULT_DB_PATH)

    try:
        # Validate and create a cached connection resource (but do not pass it into cached_data functions)
        conn = get_conn(db_path)
    except FileNotFoundError as e:
        st.sidebar.error(str(e))
        st.error("Database file not found. Place `ferc_data.db` in the repo or update the path in the sidebar.")
        return

    tables = list_tables(db_path)
    if not tables:
        st.warning("No tables found in the database.")
        return

    with st.sidebar:
        st.markdown("---")
        st.header("Table & View")
        table = st.selectbox("Table", tables)
        limit = st.number_input("Row limit", min_value=10, max_value=200000, value=1000, step=10)
        sample_percent = st.slider("Sample % (0 = full limit)", 0, 100, 0)
        show_sql = st.checkbox("Show executed SQL", value=False)
        st.markdown("---")
        st.write("Deployment: Streamlit Community Cloud or Hugging Face Spaces")

    df = read_table(db_path, table, limit=limit)
    if sample_percent:
        df = df.sample(frac=sample_percent / 100.0)

    st.subheader(f"{table} — {len(df)} rows (showing up to {limit})")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, file_name=f"{table}.csv", mime="text/csv")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        st.sidebar.markdown("### Quick chart")
        x = st.sidebar.selectbox("X (numeric)", numeric_cols, index=0)
        y = st.sidebar.selectbox("Y (numeric)", numeric_cols, index=min(1, max(0, len(numeric_cols) - 1)))
        chart = (
            alt.Chart(df.dropna(subset=[x, y]))
            .mark_circle(size=60)
            .encode(x=alt.X(x, type="quantitative"), y=alt.Y(y, type="quantitative"), tooltip=list(df.columns))
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No numeric columns available for plotting in this table.")

    if show_sql:
        st.code(f"SELECT * FROM \"{table}\" LIMIT {int(limit)}")


if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import sqlite3
import altair as alt
import os
from typing import List


st.set_page_config(page_title="FERC Data Explorer", layout="wide")

# Default DB path inside the repo; adjust if your DB is elsewhere
DEFAULT_DB = os.path.join("ferc-eqr-scraper", "ferc_data.db")


@st.cache_resource
def get_conn(db_path: str):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")
    # sqlite3 connections are not inherently thread-safe; allow check_same_thread=False
    # Use cache_resource for unserializable objects like DB connections
    return sqlite3.connect(db_path, check_same_thread=False)


@st.cache_data(ttl=600)
def list_tables(db_path: str) -> List[str]:
    q = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    with sqlite3.connect(db_path) as conn:
        return [r[0] for r in conn.execute(q).fetchall()]


@st.cache_data(ttl=600)
def read_table(db_path: str, table_name: str, limit: int = 1000) -> pd.DataFrame:
    q = f"SELECT * FROM \"{table_name}\" LIMIT {limit}"
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(q, conn)


def main():
    st.title("FERC Data Explorer")

    st.sidebar.header("Configuration")
    db_path = st.sidebar.text_input("Path to SQLite DB", value=DEFAULT_DB)
    try:
        conn = get_conn(db_path)
    except FileNotFoundError as e:
        st.sidebar.error(str(e))
        st.error("Database file not found. Place `ferc_data.db` in the repo or update the path in the sidebar.")
        return

    tables = list_tables(conn)
    if not tables:
        st.warning("No tables found in the database.")
        return

    with st.sidebar:
        st.markdown("---")
        st.header("Table & View")
        table = st.selectbox("Table", tables)
        limit = st.number_input("Row limit", min_value=10, max_value=100000, value=1000, step=10)
        sample_percent = st.slider("Sample % (0 = full limit)", 0, 100, 0)
        st.markdown("---")
        st.write("Deploy: Streamlit Community Cloud or Hugging Face Spaces")

    df = read_table(conn, table, limit=limit)
    if sample_percent:
        df = df.sample(frac=sample_percent / 100.0)

    st.subheader(f"{table} — {len(df)} rows (showing up to {limit})")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, file_name=f"{table}.csv", mime="text/csv")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        st.sidebar.markdown("### Quick chart")
        x = st.sidebar.selectbox("X (numeric)", numeric_cols, index=0)
        y = st.sidebar.selectbox("Y (numeric)", numeric_cols, index=min(1, max(0, len(numeric_cols) - 1)))
        chart = (
            alt.Chart(df.dropna(subset=[x, y]))
            .mark_circle(size=60)
            .encode(x=alt.X(x, type="quantitative"), y=alt.Y(y, type="quantitative"), tooltip=list(df.columns))
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No numeric columns available for plotting in this table.")


if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import sqlite3
import altair as alt
import os
from typing import List

st.set_page_config(page_title="FERC Data Explorer", layout="wide")

# Path to the SQLite DB inside this repo. Adjust if you place it elsewhere.
DEFAULT_DB_PATH = os.path.join("ferc-eqr-scraper", "ferc_data.db")


@st.cache_data(ttl=600)
def get_conn(db_path: str):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")
    # sqlite3 connections are not thread-safe by default; Streamlit runs in a single process
    return sqlite3.connect(db_path, check_same_thread=False)


@st.cache_data(ttl=600)
def list_tables(conn) -> List[str]:
    q = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    return [r[0] for r in conn.execute(q).fetchall()]


@st.cache_data(ttl=600)
def read_table(conn, table_name: str, limit: int = 1000) -> pd.DataFrame:
    # Basic protection against SQL injection via identifier quoting
    q = f"SELECT * FROM \"{table_name}\" LIMIT {int(limit)}"
    return pd.read_sql_query(q, conn)


def main():
    st.title("FERC Data Explorer")
    st.markdown(
        "This is a lightweight Streamlit viewer for the `ferc_data.db` SQLite database included in the repository. Use the sidebar to pick a table and visualize data."
    )

    db_path = st.text_input("Database path", value=DEFAULT_DB_PATH)

    try:
        conn = get_conn(db_path)
    except FileNotFoundError:
        st.error(f"Database file not found at: {db_path}\n\nMake sure the file exists and the path is correct. If you plan to deploy to Streamlit Cloud, place the DB in the repository before deploying.")
        st.stop()

    tables = list_tables(conn)
    if not tables:
        st.warning("No tables found in the database.")
        st.stop()

    with st.sidebar:
        st.header("Controls")
        table = st.selectbox("Table", tables)
        limit = st.number_input("Row limit", min_value=10, max_value=200000, value=1000, step=10)
        sample_percent = st.slider("Sample % (0 = full limit)", 0, 100, 0)
        show_sql = st.checkbox("Show executed SQL", value=False)
        st.markdown("---")
        st.write("Deployment: Streamlit Cloud")
        st.caption("If you deploy to Streamlit Community Cloud, include the DB file in your repo or modify the path to download from cloud storage.")

    df = read_table(conn, table, limit=limit)
    if sample_percent:
        df = df.sample(frac=sample_percent / 100.0)

    st.subheader(f"{table} — {len(df)} rows (showing up to {limit})")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, file_name=f"{table}.csv", mime="text/csv")

    # Quick chart
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        st.sidebar.markdown("### Quick chart")
        x = st.sidebar.selectbox("X (numeric)", numeric_cols, index=0)
        y = st.sidebar.selectbox("Y (numeric)", numeric_cols, index=min(1, max(0, len(numeric_cols) - 1)))
        chart = (
            alt.Chart(df.dropna(subset=[x, y]))
            .mark_circle(size=60)
            .encode(x=alt.X(x, type="quantitative"), y=alt.Y(y, type="quantitative"), tooltip=list(df.columns))
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No numeric columns available for plotting in this table.")

    if show_sql:
        st.code(f"SELECT * FROM \"{table}\" LIMIT {int(limit)}")


if __name__ == "__main__":
    main()
