import streamlit as st
import pandas as pd
import sqlite3
import altair as alt
import os
from typing import List

st.set_page_config(page_title="FERC Data Explorer", layout="wide")

# Path to the SQLite DB inside this repo
DEFAULT_DB_PATH = os.path.join("ferc-eqr-scraper", "ferc_data.db")


@st.cache_data(ttl=600)
def list_tables(db_path: str) -> List[str]:
    """List table names from SQLite database."""
    q = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    with sqlite3.connect(db_path) as conn:
        return [r[0] for r in conn.execute(q).fetchall()]


@st.cache_data(ttl=600)  
def read_table(db_path: str, table_name: str, limit: int = 1000) -> pd.DataFrame:
    """Read table data from SQLite database."""
    q = f"SELECT * FROM \"{table_name}\" LIMIT {int(limit)}"
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(q, conn)


def main():
    st.title("FERC Data Explorer")
    st.markdown("Explore the FERC SQLite database with table views, charts, and CSV downloads.")

    # Configuration
    st.sidebar.header("Configuration")
    db_path = st.sidebar.text_input("Database path", value=DEFAULT_DB_PATH)
    
    # Validate database exists
    if not os.path.exists(db_path):
        st.error(f"Database not found at: {db_path}")
        st.info("Make sure `ferc_data.db` is in the `ferc-eqr-scraper/` folder or update the path above.")
        return

    # Get tables
    try:
        tables = list_tables(db_path)
    except Exception as e:
        st.error(f"Error reading database: {e}")
        return
        
    if not tables:
        st.warning("No tables found in the database.")
        return

    # Sidebar controls
    with st.sidebar:
        st.markdown("---")
        st.header("Table Controls")
        table = st.selectbox("Select table", tables)
        limit = st.number_input("Row limit", min_value=10, max_value=50000, value=1000, step=100)
        sample_percent = st.slider("Sample % (0 = show all up to limit)", 0, 100, 0)
        show_sql = st.checkbox("Show SQL query")
        
        st.markdown("---")
        st.caption("Deploy to Streamlit Community Cloud for public access")

    # Read and display data
    try:
        df = read_table(db_path, table, limit=limit)
        
        if sample_percent > 0:
            df = df.sample(frac=sample_percent / 100.0)
            
        st.subheader(f"Table: {table}")
        st.write(f"Showing {len(df)} rows")
        
        # Display table
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download as CSV", 
            csv, 
            file_name=f"{table}.csv", 
            mime="text/csv"
        )
        
        # Show SQL if requested
        if show_sql:
            st.code(f"SELECT * FROM \"{table}\" LIMIT {int(limit)}")
            
        # Chart for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) >= 2:
            st.sidebar.markdown("### Quick Chart")
            x_col = st.sidebar.selectbox("X axis", numeric_cols, index=0)
            y_col = st.sidebar.selectbox("Y axis", numeric_cols, index=1)
            
            chart_df = df.dropna(subset=[x_col, y_col])
            if len(chart_df) > 0:
                chart = (
                    alt.Chart(chart_df)
                    .mark_circle(size=60, opacity=0.7)
                    .encode(
                        x=alt.X(x_col, type="quantitative"),
                        y=alt.Y(y_col, type="quantitative"), 
                        tooltip=[x_col, y_col]
                    )
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)
        elif len(numeric_cols) == 1:
            st.info(f"Only one numeric column ({numeric_cols[0]}) available for plotting.")
        else:
            st.info("No numeric columns available for plotting.")
            
    except Exception as e:
        st.error(f"Error reading table '{table}': {e}")


if __name__ == "__main__":
    main()
