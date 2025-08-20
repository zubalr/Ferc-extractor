import os
from typing import List, Optional

import pandas as pd
import streamlit as st
import sqlalchemy
from sqlalchemy import text
import importlib
from typing import Any
import plotly.express as px


st.set_page_config(
    page_title="FERC EQR Analytics Dashboard", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)


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
    auth_token = os.environ.get("TURSO_AUTH_TOKEN")
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
            msg = str(e)
            if url and url.startswith("libsql://"):
                extra = (
                    "Failed to connect to libsql/Turso. Common causes:\n"
                    "  - missing libsql SQLAlchemy dialect/plugin on the environment\n"
                    "  - missing TURSO_AUTH_TOKEN in environment or Streamlit secrets\n\n"
                    "How to fix:\n"
                    "  1) Add the libsql SQLAlchemy dialect and client to your requirements, e.g. in requirements.txt add:\n"
                    "       libsql\n"
                    "       sqlalchemy-libsql\n"
                    "     (Then redeploy on Streamlit Cloud so the dialect is available.)\n"
                    "  2) Ensure you have set TURSO_AUTH_TOKEN (Streamlit secrets or env) with your Turso auth token.\n"
                    "  3) Alternatively, use a different SQLAlchemy-compatible URL if available (postgresql://, mysql://, etc.).\n\n"
                    "If you want me to switch to using the libsql Python client directly (avoid SQLAlchemy dialects), I can implement that fallback — say the word and I'll add it.\n"
                )
                # If the root cause looks like missing plugin, make it explicit
                if "Can't load plugin: sqlalchemy.dialects:libsql" in msg or "NoSuchModuleError" in msg:
                    raise RuntimeError(extra + f"\nOriginal error: {msg}")
                raise RuntimeError(f"Failed to connect to libsql/Turso. Ensure TURSO_AUTH_TOKEN env var is set and the URL is correct. Original error: {msg}")
        raise


@st.cache_resource
def get_libsql_client(url: str, auth_token: Optional[str] = None) -> Any:
    """Try to create a libsql client/connection object. Return an adapter dict with type and client.
    This function uses importlib to avoid hard dependency until runtime.
    """
    try:
        libsql = importlib.import_module("libsql")
    except Exception:
        # Fall back to HTTP adapter if libsql package not installed
        return {"type": "http", "url": url, "auth_token": auth_token}

    # Prefer a Client class if present
    if hasattr(libsql, "Client"):
        Client = getattr(libsql, "Client")
        try:
            # many libsql client constructors accept url and headers
            headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else None
            client = Client(url, headers=headers) if headers is not None else Client(url)
            return {"type": "client", "client": client}
        except Exception:
            # fallback to trying connect
            pass

    # Next try a connect() style API (returns a connection object)
    if hasattr(libsql, "connect"):
        try:
            kwargs = {"headers": {"Authorization": f"Bearer {auth_token}"}} if auth_token else {}
            conn = libsql.connect(url, **kwargs)
            return {"type": "connection", "client": conn}
        except Exception:
            # fallback to http adapter
            return {"type": "http", "url": url, "auth_token": auth_token}

    # If we reach here, return HTTP adapter as a last resort
    return {"type": "http", "url": url, "auth_token": auth_token}


def _http_execute(url: str, auth_token: Optional[str], sql: str, timeout: int = 30):
    """Execute SQL against a libsql HTTP endpoint. Converts libsql://host to https://host."""
    import requests
    # derive host
    # url may be libsql://<host> or libsql://<host>/<path>
    if url.startswith("libsql://"):
        host = url[len("libsql://"):]
    else:
        host = url
    # ensure no trailing slash prefix
    # Turso/libsql HTTP endpoint expects a JSON body with a `statements` field.
    # Use the base host (no /sql) and send payload: {"statements": [{"sql": "..."}]}
    base = f"https://{host.rstrip('/')}"

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    # Preferred Turso payload shape
    payloads = [
        {"statements": [{"sql": sql}]},
        # fallback: older shapes or simpler API
        {"statements": [sql]},
        {"sql": sql},
    ]

    last_exc = None
    for payload in payloads:
        try:
            resp = requests.post(base, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            try:
                json_resp = resp.json()
                # Debug: Let's see the raw response structure
                print(f"DEBUG: Raw Turso response: {json_resp}")
                
                # Handle Turso response format which can be:
                # 1. [{"results": {"columns": [...], "rows": [...]}}] - array wrapper
                # 2. {"results": [{"columns": [...], "rows": [...]}]} - object wrapper  
                
                # Case 1: Array wrapper - extract first element
                if isinstance(json_resp, list) and len(json_resp) > 0:
                    first_item = json_resp[0]
                    print(f"DEBUG: First array item: {first_item}")
                    if isinstance(first_item, dict) and "results" in first_item:
                        # Extract the results content directly
                        results_content = first_item["results"]
                        print(f"DEBUG: Extracted results content: {results_content}")
                        return results_content
                    return first_item
                
                # Case 2: Object wrapper with results array  
                elif isinstance(json_resp, dict) and "results" in json_resp:
                    results = json_resp["results"]
                    if isinstance(results, list) and len(results) > 0:
                        first_result = results[0]
                        print(f"DEBUG: First result from array: {first_result}")
                        return first_result
                    return results
                
                return json_resp
            except Exception:
                return resp.text
        except Exception as e:
            last_exc = e
            continue

    if last_exc is not None:
        raise last_exc



def _normalize_libsql_result(res: Any) -> pd.DataFrame:
    """Normalize possible libsql client results into a pandas DataFrame.
    This attempts several common return shapes.
    """
    # If it's already a DataFrame
    if isinstance(res, pd.DataFrame):
        return res

    # Try to parse string representations of dicts (common with HTTP responses)
    if isinstance(res, str):
        try:
            import ast
            # Try to safely evaluate string representation of dict/list
            res = ast.literal_eval(res)
        except (ValueError, SyntaxError):
            # If that fails, try JSON parsing
            try:
                import json
                res = json.loads(res)
            except json.JSONDecodeError:
                # If both fail, treat as single value
                return pd.DataFrame([res], columns=['value'])

    # If it's a list of dicts
    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
        return pd.DataFrame(res)

    # If it's a list of tuples with no column info, create generic columns
    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], (list, tuple)):
        # create column names c0..cN
        cols = [f"c{i}" for i in range(len(res[0]))]
        return pd.DataFrame(res, columns=cols)

    # If it's a dict-like response with 'rows' and 'columns' keys (Turso format)
    if isinstance(res, dict):
        # Handle the specific Turso response format first
        # {"columns": ["name"], "rows": [["contacts"], ["contracts"], ...]}
        rows = res.get("rows")
        cols = res.get("columns") or res.get("cols")
        
        if rows is not None and cols is not None:
            try:
                df = pd.DataFrame(rows, columns=cols)
                return df
            except Exception as e:
                # Fallback if column count doesn't match
                try:
                    return pd.DataFrame(rows)
                except Exception:
                    pass
        
        # If only rows are present, create generic columns
        if rows is not None:
            try:
                if isinstance(rows[0], (list, tuple)) and len(rows) > 0:
                    cols = [f"col_{i}" for i in range(len(rows[0]))]
                    return pd.DataFrame(rows, columns=cols)
                return pd.DataFrame(rows)
            except Exception:
                return pd.DataFrame(rows)

        # nested results: {"results": [{"columns": [...], "rows": [...]}, ...]}
        results = res.get("results") or res.get("result") or res.get("data")
        if isinstance(results, list) and len(results) > 0:
            first = results[0]
            if isinstance(first, dict) and ("rows" in first or "columns" in first or "cols" in first):
                rows = first.get("rows")
                cols = first.get("columns") or first.get("cols")
                if rows is not None:
                    try:
                        if cols:
                            return pd.DataFrame(rows, columns=cols)
                        return pd.DataFrame(rows)
                    except Exception:
                        return pd.DataFrame(rows)
            # sometimes results is a list of dicts representing rows
            if isinstance(first, dict) and all(isinstance(v, (str, int, float, type(None))) for v in first.values()):
                try:
                    return pd.DataFrame(results)
                except Exception:
                    pass

    # Last resort: wrap single scalar
    try:
        return pd.DataFrame([res])
    except Exception:
        raise RuntimeError("Unable to normalize libsql result into DataFrame")


def libsql_list_tables(adapter: dict) -> List[str]:
    client = adapter.get("client")
    sqls = [
        "SELECT name FROM sqlite_schema WHERE type='table' ORDER BY name;",
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;",
    ]
    for sql in sqls:
        try:
            if adapter.get("type") == "client":
                res = client.execute(sql)
            elif adapter.get("type") == "connection":
                cur = client.cursor()
                cur.execute(sql)
                rows = cur.fetchall()
                res = rows
            elif adapter.get("type") == "http":
                # use http endpoint
                res = _http_execute(adapter["url"], adapter.get("auth_token"), sql)
            else:
                continue
            
            # CRITICAL: Always normalize the result to extract table names
            df = _normalize_libsql_result(res)
            
            if df is not None and not df.empty:
                # try to find a name column
                if "name" in df.columns:
                    names = [str(x) for x in df["name"].tolist()]
                    # Filter out system tables and return only user tables
                    return [name for name in names if not name.startswith('sqlite_')]
                # else take first column
                elif df.shape[1] >= 1:
                    names = [str(x) for x in df.iloc[:, 0].tolist()]
                    return [name for name in names if not name.startswith('sqlite_')]
        except Exception:
            continue
    return []


def libsql_table_row_count(adapter: dict, table_name: str) -> int:
    client = adapter.get("client")
    sql = f'SELECT COUNT(*) as cnt FROM "{table_name}"'
    try:
        if adapter.get("type") == "client":
            res = client.execute(sql)
        elif adapter.get("type") == "connection":
            cur = client.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            res = rows
        elif adapter.get("type") == "http":
            res = _http_execute(adapter["url"], adapter.get("auth_token"), sql)
        else:
            return 0
        
        df = _normalize_libsql_result(res)
        # expected single value
        if df is not None and df.size > 0:
            return int(df.iat[0, 0])
    except Exception as e:
        print(f"Error counting rows in table {table_name}: {e}")
        pass
    return 0


def libsql_read_table(adapter: dict, table_name: str, limit: int = 1000) -> pd.DataFrame:
    client = adapter.get("client")
    sql = f'SELECT * FROM "{table_name}" LIMIT {int(limit)}'
    try:
        if adapter.get("type") == "client":
            res = client.execute(sql)
        elif adapter.get("type") == "connection":
            cur = client.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            # try to get column names
            cols = None
            try:
                cols = [d[0] for d in cur.description]
            except Exception:
                pass
            if cols is not None:
                return pd.DataFrame(rows, columns=cols)
            res = rows
        elif adapter.get("type") == "http":
            res = _http_execute(adapter["url"], adapter.get("auth_token"), sql)
        else:
            raise RuntimeError("Unknown libsql adapter type")
        
        df = _normalize_libsql_result(res)
        # Ensure we return a proper DataFrame, even if empty
        if df is None:
            return pd.DataFrame()
        return df
    except Exception as e:
        # Return empty DataFrame instead of raising to avoid UI breaking
        print(f"Error reading table {table_name}: {e}")
        return pd.DataFrame()


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


def get_connection_info():
    """Get database connection info automatically from environment."""
    env_turso = os.environ.get("TURSO_DATABASE_URL")
    auth_token = os.environ.get("TURSO_AUTH_TOKEN")
    
    if not env_turso:
        st.error("⚠️ Database connection not configured. Please set TURSO_DATABASE_URL in Streamlit secrets.")
        st.stop()
    
    if not auth_token:
        st.error("⚠️ Database authentication not configured. Please set TURSO_AUTH_TOKEN in Streamlit secrets.")
        st.stop()
        
    return env_turso


def create_dashboard_header():
    """Create a beautiful header for the dashboard."""
    # Main header with custom styling
    st.markdown("""
        <div style="
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h1 style="
                color: white;
                text-align: center;
                margin: 0;
                font-size: 2.5rem;
                font-weight: 700;
            ">⚡ FERC EQR Analytics Dashboard</h1>
            <p style="
                color: rgba(255, 255, 255, 0.9);
                text-align: center;
                margin: 0.5rem 0 0 0;
                font-size: 1.2rem;
            ">Federal Energy Regulatory Commission • Electric Quarterly Reports</p>
        </div>
    """, unsafe_allow_html=True)


def create_sidebar_filters():
    """Create modern sidebar with filters and controls."""
    st.sidebar.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #667eea; margin: 0;">🎛️ Dashboard Controls</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Data settings
    st.sidebar.markdown("### 📊 Data Settings")
    preview_limit = st.sidebar.slider(
        "Table Preview Limit",
        min_value=100, 
        max_value=10000, 
        value=1000,
        step=100,
        help="Maximum number of rows to load for analysis"
    )
    
    sample_percent = st.sidebar.slider(
        "Sample Percentage", 
        min_value=0, 
        max_value=100, 
        value=20,
        help="Percentage of data to sample for faster analysis"
    )
    
    # Analysis options
    st.sidebar.markdown("### 🔍 Analysis Options")
    show_advanced = st.sidebar.checkbox("Advanced Analytics", value=True)
    show_sql = st.sidebar.checkbox("Show SQL Queries", value=False)
    auto_refresh = st.sidebar.checkbox("Auto-refresh Data", value=False)
    
    if auto_refresh:
        refresh_interval = st.sidebar.selectbox(
            "Refresh Interval",
            [30, 60, 300, 600],
            index=1,
            format_func=lambda x: f"{x//60} min" if x >= 60 else f"{x} sec"
        )
    else:
        refresh_interval = None
    
    # Export options
    st.sidebar.markdown("### 💾 Export Options")
    export_format = st.sidebar.selectbox(
        "Export Format",
        ["CSV", "Excel", "JSON", "Parquet"],
        index=0
    )
    
    return preview_limit, sample_percent, show_advanced, show_sql, auto_refresh, refresh_interval, export_format


def main():
    # Create beautiful dashboard header
    create_dashboard_header()
    
    # Get sidebar controls
    preview_limit, sample_percent, show_advanced, show_sql, auto_refresh, refresh_interval, export_format = create_sidebar_filters()
    
    # Get connection info automatically
    db_input = get_connection_info()
    
    # Create database connection
    libsql_adapter = None
    try:
        auth_token = os.environ.get("TURSO_AUTH_TOKEN")
        libsql_adapter = get_libsql_client(db_input, auth_token=auth_token)
        if not isinstance(libsql_adapter, dict) or "type" not in libsql_adapter:
            st.error(f"❌ Database adapter error: {libsql_adapter!r}")
            st.stop()
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        st.stop()

    # Get tables
    try:
        tables = libsql_list_tables(libsql_adapter)
    except Exception as e:
        st.error(f"❌ Failed to retrieve database schema: {e}")
        st.stop()

    if not tables:
        st.warning("⚠️ No tables found in the database.")
        st.stop()

    # Database overview metrics
    create_database_overview(libsql_adapter, tables)
    
    st.markdown("---")
    
    # Main dashboard layout
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        create_table_selector(tables, libsql_adapter, preview_limit, sample_percent, show_sql, export_format)
    
    with col2:
        selected_table = st.session_state.get('selected_table', tables[0])
        if selected_table:
            create_data_explorer(libsql_adapter, selected_table, preview_limit, sample_percent, show_advanced, show_sql)


def create_database_overview(libsql_adapter, tables):
    """Create a beautiful database overview section."""
    st.markdown("## 📊 Database Overview")
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate total rows across all tables (sample first 5 for performance)
    total_rows = 0
    for table in tables[:5]:
        try:
            count = libsql_table_row_count(libsql_adapter, table)
            total_rows += count
        except Exception:
            pass
    
    # Display metrics with beautiful styling
    with col1:
        st.markdown("""
            <div style="
                background: linear-gradient(45deg, #FF6B6B, #FF8E8E);
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <h3 style="margin: 0; font-size: 2rem;">📋</h3>
                <h4 style="margin: 0;">{}</h4>
                <p style="margin: 0; opacity: 0.9;">Tables</p>
            </div>
        """.format(len(tables)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="
                background: linear-gradient(45deg, #4ECDC4, #44A08D);
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <h3 style="margin: 0; font-size: 2rem;">📊</h3>
                <h4 style="margin: 0;">{:,}</h4>
                <p style="margin: 0; opacity: 0.9;">Total Rows</p>
            </div>
        """.format(total_rows), unsafe_allow_html=True)
    
    with col3:
        # Get adapter type
        adapter_type = libsql_adapter.get("type", "unknown").title()
        st.markdown("""
            <div style="
                background: linear-gradient(45deg, #A8E6CF, #7FCDCD);
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <h3 style="margin: 0; font-size: 2rem;">🔗</h3>
                <h4 style="margin: 0;">{}</h4>
                <p style="margin: 0; opacity: 0.9;">Connection</p>
            </div>
        """.format(adapter_type), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div style="
                background: linear-gradient(45deg, #667eea, #764ba2);
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <h3 style="margin: 0; font-size: 2rem;">⚡</h3>
                <h4 style="margin: 0;">Live</h4>
                <p style="margin: 0; opacity: 0.9;">Status</p>
            </div>
        """, unsafe_allow_html=True)

    with left:
        st.header("Explore Tables")
        if not tables:
            st.info("No tables discovered. If your Turso DB contains tables, ensure the TURSO_AUTH_TOKEN is correct and redeploy.")
            return
        
        # Set default table (avoid transactions, prefer first non-transactions table)
        default_table = tables[0]  # fallback to first
        for t in tables:
            if t.lower() not in ['transactions', 'transaction']:
                default_table = t
                break
        
        # Find the index of the default table
        try:
            default_index = tables.index(default_table)
        except ValueError:
            default_index = 0
            
        table = st.selectbox("Select table", tables, index=default_index)
        show_sql = st.checkbox("Show SQL preview")
        sample_percent = st.slider("Sample % (0 = none)", 0, 100, 0)  # Default to 0% sampling
        download_all = st.button("Download full table CSV")

        st.markdown("---")
        st.caption("Table stats and quick filters appear here after selecting a table.")

    # Load table data
    # Ensure `table` is a simple string
    if not isinstance(table, str):
        try:
            table = str(table)
        except Exception:
            st.error("Selected table name is not a string. Aborting.")
            return

    try:
        if libsql_adapter is not None:
            df = libsql_read_table(libsql_adapter, table, limit=preview_limit)
        else:
            df = read_table(engine, table, limit=preview_limit)
            
    except Exception as e:
        st.error(f"Error reading table '{table}': {e}")
        return

    if sample_percent > 0 and sample_percent < 100 and not df.empty:
        try:
            df = df.sample(frac=sample_percent / 100.0)
        except Exception as e:
            st.sidebar.warning(f"Could not apply sampling: {e}")

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
                if libsql_adapter is not None:
                    full_df = libsql_read_table(libsql_adapter, table, limit=10**9)
                else:
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
                # Use a safe conversion to DataFrame to avoid duplicate column names
                # Rename the series axis to the category name, then reset_index with a count column
                try:
                    vc_df = vc.rename_axis(cat).reset_index(name="count")
                except Exception:
                    # Fallback: generic column names
                    vc_df = vc.reset_index()
                    vc_df.columns = ["category", "count"]
                    cat = "category"

                fig2 = px.bar(vc_df, x=cat, y="count", title=f"Top {topn}: {cat}")
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
                miss_df = miss.reset_index()
                miss_df.columns = ["column", "missing_frac"]
                fig_miss = px.bar(miss_df, x="column", y="missing_frac", title="Missing fraction by column")
                # Plotly expects a named column
                if fig_miss.data:
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
                    if libsql_adapter is not None:
                        full = libsql_read_table(libsql_adapter, table, limit=10**9)
                    else:
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
