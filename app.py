import os
from typing import List, Optional

import pandas as pd
import streamlit as st
import sqlalchemy
from sqlalchemy import text
import importlib
from typing import Any
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
                
                # For Turso API, extract the first result from the response
                if isinstance(json_resp, dict) and "results" in json_resp:
                    results = json_resp["results"]
                    if isinstance(results, list) and len(results) > 0:
                        first_result = results[0]
                        print(f"DEBUG: First result: {first_result}")
                        return first_result  # Return first result, not the wrapper
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
    print(f"DEBUG: Normalizing result type: {type(res)}")
    print(f"DEBUG: Normalizing result content: {res}")
    
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
        print(f"DEBUG: Processing dict with keys: {list(res.keys())}")
        
        # Handle the specific Turso response format first
        # {"columns": ["name"], "rows": [["contacts"], ["contracts"], ...]}
        rows = res.get("rows")
        cols = res.get("columns") or res.get("cols")
        
        print(f"DEBUG: Found rows: {rows}")
        print(f"DEBUG: Found columns: {cols}")
        
        if rows is not None and cols is not None:
            try:
                df = pd.DataFrame(rows, columns=cols)
                print(f"DEBUG: Successfully created DataFrame with shape: {df.shape}")
                return df
            except Exception as e:
                print(f"DEBUG: Failed to create DataFrame with columns: {e}")
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


def sidebar_connection_controls():
    st.sidebar.header("Connection")
    st.sidebar.caption("This app connects to Turso via TURSO_DATABASE_URL and TURSO_AUTH_TOKEN set in Streamlit secrets.")
    # Only allow Turso URL (from env) — user asked to remove generic DATABASE_URL
    env_turso = os.environ.get("TURSO_DATABASE_URL")
    if env_turso:
        st.sidebar.write("Turso URL:")
        st.sidebar.code(env_turso)
    else:
        st.sidebar.error("TURSO_DATABASE_URL not set in environment/secrets. Please add it in Streamlit Cloud secrets.")

    allow_local = st.sidebar.checkbox("Allow using local repo sqlite (not recommended)")
    limit = st.sidebar.number_input("Row limit (table preview)", value=2000, min_value=100, max_value=200000, step=100)
    return env_turso or "", limit, allow_local


def main():
    st.title("FERC Explorer")
    st.markdown("A modern, interactive dashboard to explore FERC EQR data.\n\nUse the sidebar to connect to your DB (set `DATABASE_URL` in Streamlit Cloud).")

    db_input, preview_limit, allow_local = sidebar_connection_controls()

    # If there's no DB URL and local is not allowed, show an error and instructions
    if not db_input and not allow_local:
        st.error("No `DATABASE_URL` provided. Please set `DATABASE_URL` in Streamlit Cloud or enter a connection string in the sidebar.\n\nDo NOT rely on the repository sqlite for deployment — enable 'Allow using local repo sqlite' only for local testing.")
        return

    # Create engine or libsql adapter
    engine = None
    libsql_adapter = None

    # If the URL is a libsql (Turso) URL, prefer using libsql client / HTTP adapter directly
    if db_input and db_input.startswith("libsql://"):
        auth_token = os.environ.get("TURSO_AUTH_TOKEN")
        try:
            # Try to get a libsql client; this may return a client adapter or an http adapter
            libsql_adapter = get_libsql_client(db_input, auth_token=auth_token)
            if not isinstance(libsql_adapter, dict) or "type" not in libsql_adapter:
                st.error(f"Libsql adapter returned unexpected value: {libsql_adapter!r}")
                return
        except Exception as e:
            st.error(f"Unable to initialize libsql adapter: {e}")
            return
    else:
        # Non-libsql DBs use SQLAlchemy engine
        try:
            engine = get_engine(db_input if db_input else DEFAULT_DB_PATH, allow_local=allow_local)
        except Exception as e:
            st.error(f"Unable to connect to the database: {e}")
            return

    # Get tables
    try:
        if libsql_adapter is not None:
            tables = libsql_list_tables(libsql_adapter)
        else:
            tables = list_tables(engine)
    except Exception as e:
        st.error(f"Error reading database schema: {e}")
        return

    # Ensure we have a list of strings
    if not isinstance(tables, list):
        st.error(f"Expected list of tables, got: {type(tables)} - {tables}")
        return
        
    # Filter to only string table names and ensure they're actually table names, not dict representations
    clean_tables = []
    for t in tables:
        if isinstance(t, str):
            # Check if it looks like a dict representation string
            if t.startswith('{') and 'columns' in t and 'rows' in t:
                # This is a stringified dict response - we need to parse it
                try:
                    import ast
                    dict_data = ast.literal_eval(t)
                    if isinstance(dict_data, dict) and 'rows' in dict_data:
                        # Extract table names from the rows
                        for row in dict_data['rows']:
                            if isinstance(row, list) and len(row) > 0:
                                table_name = str(row[0])
                                if not table_name.startswith('sqlite_'):
                                    clean_tables.append(table_name)
                except Exception:
                    # If parsing fails, skip this item
                    continue
            else:
                # Regular table name
                if not t.startswith('sqlite_'):
                    clean_tables.append(t)
    
    tables = clean_tables

    if not tables:
        # If HTTP adapter, probe endpoint for debugging
        if libsql_adapter is not None and libsql_adapter.get("type") == "http":
            # Try both /sql and root endpoints and show full diagnostic info
            host = libsql_adapter.get("url")
            auth = libsql_adapter.get("auth_token")
            base = host[len("libsql://"):] if host.startswith("libsql://") else host
            base = base.rstrip('/')
            ep = f"https://{base}"
            probe_results = []
            try:
                import requests
                headers = {"Content-Type": "application/json", "Accept": "application/json"}
                if auth:
                    headers["Authorization"] = f"Bearer {auth}"
                # Use Turso expected payload shape
                payload = {"statements": [{"sql": "SELECT 1"}]}
                resp = requests.post(ep, json=payload, headers=headers, timeout=15)
                try:
                    body = resp.json()
                except Exception:
                    body = resp.text
                probe_results.append({"endpoint": ep, "status_code": resp.status_code, "body": body, "headers": dict(resp.headers)})
            except Exception as e:
                probe_results.append({"endpoint": ep, "error": str(e)})

            st.warning("No tables found in the database. Probe results below for debugging (attempting base host with Turso 'statements' payload).")
            for pr in probe_results:
                st.write(pr)
        else:
            st.warning("No tables found in the database.")
        return

    # Determine adapter type for display
    adapter_type = "sqlalchemy" if engine is not None else (libsql_adapter.get("type") if isinstance(libsql_adapter, dict) else "unknown")

    # Top row: quick stats
    with st.container():
        col1, col2, col3 = st.columns([2, 3, 2])
        col1.metric("Connected DB", os.path.basename(db_input) if db_input else "(default)")
        col1.write(f"Adapter: {adapter_type}")
        col2.metric("Tables", len(tables))
        # estimate total rows across first 5 tables (fast)
        sample_tables = tables[:5]
        total_preview_rows = 0
        for t in sample_tables:
            try:
                if libsql_adapter is not None:
                    total_preview_rows += libsql_table_row_count(libsql_adapter, t)
                else:
                    total_preview_rows += table_row_count(engine, t)
            except Exception:
                pass
        col3.metric("Rows (sample)", f"{total_preview_rows}")

    st.markdown("---")

    # Main layout: left side controls, right side content
    left, right = st.columns([1, 3])

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
        sample_percent = st.slider("Sample % (0 = none)", 0, 100, 10)  # Default to 10% sampling
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
        
        # Debug: Show what we got
        st.sidebar.write(f"DEBUG: Loaded {len(df)} rows, {len(df.columns) if not df.empty else 0} columns")
        if df.empty:
            st.sidebar.error("DEBUG: DataFrame is empty!")
        else:
            st.sidebar.write(f"DEBUG: Columns: {list(df.columns)}")
            
    except Exception as e:
        st.error(f"Error reading table '{table}': {e}")
        return

    if sample_percent > 0 and sample_percent < 100 and not df.empty:
        try:
            df = df.sample(frac=sample_percent / 100.0)
            st.sidebar.write(f"DEBUG: Applied {sample_percent}% sampling, now {len(df)} rows")
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
