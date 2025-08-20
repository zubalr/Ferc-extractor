import os
from typing import List, Optional

import pandas as pd
import streamlit as st
import sqlalchemy
from sqlalchemy import text
import plotly.express as px


st.set_page_config(page_title="FERC Explorer — Modern Dashboard", layout="wide")


# Default path (local file) — but we will not use it unless explicitly allowed by user
DEFAULT_DB_PATH = os.path.join("ferc-eqr-scraper", "ferc_data.db")


@st.cache_resource
def get_engine(db_url_or_path: Optional[str] = None, allow_local: bool = False):
    """Create a SQLAlchemy engine. Prefer a SQLAlchemy URL (DATABASE_URL). If allow_local is True and the
    provided path exists as a file, convert to sqlite URL. Otherwise raise on bad connection.
    """
    if not db_url_or_path:
        raise ValueError("No database URL provided. Set DATABASE_URL in Streamlit Cloud or provide a connection string.")

    # If looks like a sqlite file path and user allowed local override
    if allow_local and os.path.exists(db_url_or_path) and not db_url_or_path.startswith("sqlite:"):
        url = f"sqlite:///{os.path.abspath(db_url_or_path)}"
    else:
        url = db_url_or_path

    # If using Turso/libsql URL (libsql://...), libsql-python expects the auth token provided via connect_args
    connect_args = {}
    # Accept either TURSO_AUTH_TOKEN or TURSO_AUTH_TOKE (typo in .env) as user might have
    auth_token = os.environ.get("TURSO_AUTH_TOKEN") or os.environ.get("TURSO_AUTH_TOKE")
    if url and url.startswith("libsql://"):
        if auth_token:
            # libsql driver can accept a header param via connect_args; SQLAlchemy dialects vary.
            # libsql-python uses `headers` in connect args (a dict of str->str). We'll pass Authorization header.
            connect_args = {"headers": {"Authorization": f"Bearer {auth_token}"}}
        else:
            # Token missing — warn caller but continue to attempt connection (the engine creation may still fail)
            st.warning("Connecting to a libsql (Turso) URL but TURSO_AUTH_TOKEN not found in environment. Add your token in Streamlit Cloud secrets as TURSO_AUTH_TOKEN.")

    try:
        engine = sqlalchemy.create_engine(url, connect_args=connect_args)
        # quick test connection; let exceptions bubble to caller
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        # Provide extra guidance for libsql URLs
        if url and url.startswith("libsql://"):
            raise RuntimeError(f"Failed to connect to libsql/Turso. Ensure TURSO_AUTH_TOKEN env var is set and the URL is correct. Original error: {e}")
        raise


@st.cache_data(ttl=300)
def list_tables(engine) -> List[str]:
    """Return list of table names from the database connected via engine."""
    inspector = sqlalchemy.inspect(engine)
    return sorted(inspector.get_table_names())


@st.cache_data(ttl=300)
def table_row_count(engine, table_name: str) -> int:
    q = text(f"SELECT COUNT(*) as cnt FROM \"{table_name}\"")
    with engine.connect() as conn:
        r = conn.execute(q).mappings().first()
        return int(r["cnt"]) if r else 0


@st.cache_data(ttl=300)
def read_table(engine, table_name: str, limit: int = 1000) -> pd.DataFrame:
    q = f'SELECT * FROM "{table_name}" LIMIT {int(limit)}'
    with engine.connect() as conn:
        return pd.read_sql_query(q, conn)


def infer_date_columns(df: pd.DataFrame) -> List[str]:
    candidates = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            candidates.append(col)
        else:
            # heuristics: name contains date or year or period
            name = col.lower()
            if any(k in name for k in ("date", "year", "period", "time")):
                candidates.append(col)
    return candidates


def sidebar_connection_controls():
    st.sidebar.header("Connection")
    st.sidebar.caption("Provide a DATABASE_URL in Streamlit Cloud (recommended). Local sqlite is disabled by default.")
    # Prefer Turso env vars if present
    env_turso = os.environ.get("TURSO_DATABASE_URL")
    env_db = os.environ.get("DATABASE_URL")
    default_val = env_turso or env_db or ""
    db_url = st.sidebar.text_input("Database URL (leave blank to use env)", value=default_val)
    allow_local = st.sidebar.checkbox("Allow using local repo sqlite (not recommended)")
    limit = st.sidebar.number_input("Row limit (table preview)", value=2000, min_value=100, max_value=200000, step=100)
    return db_url or default_val, limit, allow_local


def main():
    st.title("FERC Explorer")
    st.markdown("A modern, interactive dashboard to explore FERC EQR data.\n\nUse the sidebar to connect to your DB (set `DATABASE_URL` in Streamlit Cloud).")

    db_input, preview_limit, allow_local = sidebar_connection_controls()

    # If there's no DB URL and local is not allowed, show an error and instructions
    if not db_input and not allow_local:
        st.error("No `DATABASE_URL` provided. Please set `DATABASE_URL` in Streamlit Cloud or enter a connection string in the sidebar.\n\nDo NOT rely on the repository sqlite for deployment — enable 'Allow using local repo sqlite' only for local testing.")
        return

    # Create engine
    try:
        engine = get_engine(db_input if db_input else DEFAULT_DB_PATH, allow_local=allow_local)
    except Exception as e:
        st.error(f"Unable to connect to the database: {e}")
        return

    # Get tables
    try:
        tables = list_tables(engine)
    except Exception as e:
        st.error(f"Error reading database schema: {e}")
        return

    if not tables:
        st.warning("No tables found in the database.")
        return

    # Top row: quick stats
    with st.container():
        col1, col2, col3 = st.columns([2, 3, 2])
        col1.metric("Connected DB", os.path.basename(db_input) if db_input else "(default)")
        col2.metric("Tables", len(tables))
        # estimate total rows across first 5 tables (fast)
        sample_tables = tables[:5]
        total_preview_rows = 0
        for t in sample_tables:
            try:
                total_preview_rows += table_row_count(engine, t)
            except Exception:
                pass
        col3.metric("Rows (sample)", f"{total_preview_rows}")

    st.markdown("---")

    # Main layout: left side controls, right side content
    left, right = st.columns([1, 3])

    with left:
        st.header("Explore Tables")
        table = st.selectbox("Select table", tables)
        show_sql = st.checkbox("Show SQL preview")
        sample_percent = st.slider("Sample % (0 = none)", 0, 100, 0)
        download_all = st.button("Download full table CSV")

        st.markdown("---")
        st.caption("Table stats and quick filters appear here after selecting a table.")

    # Load table data
    try:
        df = read_table(engine, table, limit=preview_limit)
    except Exception as e:
        st.error(f"Error reading table '{table}': {e}")
        return

    if sample_percent > 0 and sample_percent < 100:
        df = df.sample(frac=sample_percent / 100.0)

    with right:
        st.subheader(f"{table} — preview {len(df)} rows")

        # Show SQL
        if show_sql:
            st.code(f'SELECT * FROM "{table}" LIMIT {int(preview_limit)}')

        # Column information
        with st.expander("Columns & Types", expanded=False):
            dtypes = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})
            st.dataframe(dtypes, use_container_width=True)

        # Basic table display with filters
        text_filter = st.text_input("Search (applies to string columns)")
        if text_filter:
            str_cols = df.select_dtypes(include=["object"]).columns.tolist()
            if str_cols:
                mask = pd.Series(False, index=df.index)
                for c in str_cols:
                    mask = mask | df[c].astype(str).str.contains(text_filter, case=False, na=False)
                df = df[mask]

        st.dataframe(df, use_container_width=True)

        # Download preview
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download preview CSV", csv, file_name=f"{table}_preview.csv", mime="text/csv")

        # Handle full download (streaming large tables isn't ideal but provide link)
        if download_all:
            try:
                full_df = pd.read_sql_table(table, con=engine)
                st.download_button("📥 Download full CSV", full_df.to_csv(index=False).encode("utf-8"), file_name=f"{table}.csv")
            except Exception as e:
                st.error(f"Failed to load full table for download: {e}")

        # Visualizations + Insights
        st.markdown("---")
        st.markdown("### Visualizations & Insights")

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        date_cols = infer_date_columns(df)

        # Small charting area
        c1, c2 = st.columns(2)

        with c1:
            if numeric_cols:
                num = st.selectbox("Numeric column (histogram)", numeric_cols, key="hist_num")
                fig = px.histogram(df, x=num, nbins=50, title=f"Distribution: {num}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns for histogram")

        with c2:
            if cat_cols:
                cat = st.selectbox("Categorical column (top values)", cat_cols, key="bar_cat")
                topn = st.slider("Top N categories", min_value=3, max_value=50, value=10, key="topn")
                vc = df[cat].fillna("(null)").value_counts().nlargest(topn)
                fig2 = px.bar(vc.reset_index().rename(columns={"index": cat, cat: "count"}), x=cat, y="count", title=f"Top {topn}: {cat}")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No categorical columns for bar chart")

        # Time series if date column present and numeric measure selected
        if date_cols and numeric_cols:
            st.markdown("---")
            st.markdown("#### Time series (if available)")
            dcol = st.selectbox("Date column", date_cols, key="date_col")
            mcol = st.selectbox("Measure (numeric)", numeric_cols, key="measure_col")
            try:
                time_df = df.copy()
                time_df[dcol] = pd.to_datetime(time_df[dcol], errors="coerce")
                ts = (
                    time_df.dropna(subset=[dcol, mcol])
                    .groupby(pd.Grouper(key=dcol, freq="M"))[mcol]
                    .sum()
                    .reset_index()
                )
                if not ts.empty:
                    fig3 = px.line(ts, x=dcol, y=mcol, title=f"{mcol} over time")
                    st.plotly_chart(fig3, use_container_width=True)
            except Exception as e:
                st.info(f"Could not build time series: {e}")

        # Insights panel: missingness, unique counts, descriptive stats, correlation heatmap
        st.markdown("---")
        st.markdown("## Insights")
        i1, i2 = st.columns([2, 3])

        with i1:
            st.markdown("### Missingness")
            miss = df.isna().mean().sort_values(ascending=False)
            if not miss.empty:
                fig_miss = px.bar(miss.reset_index().rename(columns={"index": "column", 0: "missing_frac"}), x="column", y=0, labels={0: "missing_frac"}, title="Missing fraction by column")
                # Plotly expects a named column
                fig_miss.data[0].name = "missing"
                st.plotly_chart(fig_miss, use_container_width=True)
            else:
                st.info("No missingness detected or empty preview.")

            st.markdown("### Top unique counts")
            uq = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
            uq_df = pd.DataFrame(list(uq.items()), columns=["column", "unique_count"]).sort_values("unique_count", ascending=False).head(25)
            st.dataframe(uq_df, use_container_width=True)

        with i2:
            st.markdown("### Summary statistics (numeric)")
            if numeric_cols:
                stats = df[numeric_cols].describe().T
                st.dataframe(stats, use_container_width=True)

                st.markdown("### Correlation heatmap")
                corr = df[numeric_cols].corr()
                if not corr.empty:
                    fig_corr = px.imshow(corr, text_auto=True, title="Correlation matrix")
                    st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("No numeric columns for descriptive stats or correlation.")

        # Button to compute heavier full-table stats via SQL (dangerous for huge tables)
        st.markdown("---")
        st.markdown("### Full-table stats (optional)")
        st.info("If you click 'Compute full-table stats' the app will query the entire table. This may be slow or expensive. Use with caution.")
        if st.button("Compute full-table stats"):
            try:
                with st.spinner("Computing full-table stats..."):
                    full = pd.read_sql_table(table, con=engine)
                    st.success(f"Loaded full table: {len(full)} rows")
                    st.markdown("**Full-table missingness (top 25)**")
                    fm = full.isna().mean().sort_values(ascending=False).head(25)
                    st.dataframe(fm.reset_index().rename(columns={"index": "column", 0: "missing_frac"}), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to compute full-table stats: {e}")

    st.markdown("---")
    st.caption("Built with ❤️ — connect via a SQLAlchemy-compatible `DATABASE_URL` or place a local sqlite at `ferc-eqr-scraper/ferc_data.db`.")


if __name__ == "__main__":
    main()
