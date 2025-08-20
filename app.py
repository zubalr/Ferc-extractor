"""
FERC EQR Analytics Dashboard
A modern Streamlit dashboard for exploring FERC EQR data with Turso database integration.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# === CONFIGURATION ===
DEFAULT_DB_PATH = "ferc-eqr-scraper/ferc_data.db"

# Page configuration
st.set_page_config(
    page_title="FERC EQR Analytics Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# === DATABASE CONNECTION UTILITIES ===

@st.cache_resource
def get_engine(db_url, allow_local=False):
    """Create a SQLAlchemy engine."""
    if not db_url and allow_local:
        db_url = f"sqlite:///{DEFAULT_DB_PATH}"
    
    if not db_url:
        raise ValueError("No database URL provided")
    
    try:
        engine = create_engine(db_url)
        # Test connection
        with engine.connect() as conn:
            pass
        return engine
    except Exception as e:
        raise SQLAlchemyError(f"Failed to connect to database: {e}")


@st.cache_resource
def get_libsql_client(url, auth_token=None):
    """Get a database client adapter."""
    
    # Handle regular SQLite URLs with standard SQLAlchemy
    if url.startswith("sqlite://"):
        try:
            engine = create_engine(url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {"type": "sqlalchemy", "engine": engine}
        except Exception as e:
            raise Exception(f"Failed to connect to SQLite database: {e}")
    
    # Handle Turso URLs with HTTP adapter
    if url.startswith("libsql://"):
        if not auth_token:
            raise Exception("Turso auth token is required for libsql:// URLs")
        
        # Test the connection by trying a simple query
        try:
            result = _http_execute(url, auth_token, "SELECT 1 as test")
            # If we get here, the connection works
            return {"type": "http", "url": url, "auth_token": auth_token}
        except Exception as e:
            raise Exception(f"Failed to connect to Turso database: {e}")
    
    # For other URLs, try standard SQLAlchemy
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"type": "sqlalchemy", "engine": engine}
    except Exception as e:
        raise Exception(f"Failed to connect to database: {e}")


def _http_execute(url, auth_token, sql):
    """Execute SQL via HTTP API using Turso's REST API."""
    import requests
    
    # Extract hostname from libsql URL
    base = url[len("libsql://"):] if url.startswith("libsql://") else url
    base = base.rstrip('/')
    
    # Use Turso's REST API endpoint
    endpoint = f"https://{base}"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }
    
    # Use Turso's correct REST API format
    payload = {
        "statements": [
            {
                "q": sql,
                "params": []
            }
        ]
    }
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
        
        # Debug the response
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        data = response.json()
        
        # Handle Turso's response format: [{"results": {...}}]
        if isinstance(data, list) and len(data) > 0:
            first_result = data[0]
            if "results" in first_result:
                return first_result["results"]
            return first_result
        elif isinstance(data, dict):
            if "results" in data:
                return data["results"]
            return data
        else:
            return {"columns": [], "rows": []}
            
    except Exception as e:
        raise Exception(f"Turso API request failed: {e}")


def _normalize_libsql_result(result):
    """Normalize libsql result to standard format."""
    if isinstance(result, dict):
        if "columns" in result and "rows" in result:
            return result
        elif "results" in result:
            return _normalize_libsql_result(result["results"])
    elif isinstance(result, list):
        if len(result) > 0:
            return _normalize_libsql_result(result[0])
    
    return {"columns": [], "rows": []}


# === DATABASE OPERATIONS ===

@st.cache_data(ttl=300)
def libsql_list_tables(_adapter):
    """List tables using database adapter."""
    sql = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    
    if _adapter["type"] == "sqlalchemy":
        inspector = inspect(_adapter["engine"])
        return inspector.get_table_names()
    elif _adapter["type"] == "http":
        result = _http_execute(_adapter["url"], _adapter["auth_token"], sql)
        normalized = _normalize_libsql_result(result)
        return [row[0] for row in normalized.get("rows", [])]
    
    raise Exception(f"Unknown adapter type: {_adapter['type']}")


@st.cache_data(ttl=60)
def libsql_table_row_count(_adapter, table_name):
    """Get row count for a table."""
    sql = f"SELECT COUNT(*) FROM `{table_name}`"
    
    if _adapter["type"] == "sqlalchemy":
        with _adapter["engine"].connect() as conn:
            result = conn.execute(text(sql))
            return result.scalar()
    elif _adapter["type"] == "http":
        result = _http_execute(_adapter["url"], _adapter["auth_token"], sql)
        normalized = _normalize_libsql_result(result)
        return normalized.get("rows", [[0]])[0][0]
    
    raise Exception(f"Unknown adapter type: {_adapter['type']}")


@st.cache_data(ttl=300)
def libsql_read_table(_adapter, table_name, limit=1000):
    """Read table data using database adapter."""
    sql = f"SELECT * FROM `{table_name}`"
    if limit:
        sql += f" LIMIT {limit}"
    
    if _adapter["type"] == "sqlalchemy":
        return pd.read_sql_query(sql, _adapter["engine"])
    elif _adapter["type"] == "http":
        result = _http_execute(_adapter["url"], _adapter["auth_token"], sql)
        normalized = _normalize_libsql_result(result)
        columns = normalized.get("columns", [])
        rows = normalized.get("rows", [])
        return pd.DataFrame(rows, columns=columns)
    
    raise Exception(f"Unknown adapter type: {_adapter['type']}")


# === UI COMPONENTS ===

def get_connection_info():
    """Get database connection info from environment."""
    turso_url = os.environ.get("TURSO_DATABASE_URL")
    turso_token = os.environ.get("TURSO_AUTH_TOKEN")
    
    if turso_url and turso_token:
        # Extract database name from Turso URL
        db_name = "Unknown"
        try:
            if "libsql://" in turso_url:
                # Extract name from libsql://name.region.turso.io format
                host_part = turso_url.replace("libsql://", "")
                db_name = host_part.split('.')[0]
        except:
            pass
            
        return {
            'url': turso_url,
            'using_turso': True,
            'db_name': db_name
        }
    
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return {
            'url': database_url,
            'using_turso': False,
            'db_name': 'Custom Database'
        }
    
    # Local database fallback
    if os.path.exists(DEFAULT_DB_PATH):
        return {
            'url': f"sqlite:///{DEFAULT_DB_PATH}",
            'using_turso': False,
            'db_name': 'Local SQLite'
        }
    
    return None


def create_dashboard_header():
    """Create an interactive dashboard header with user guidance."""
    st.markdown("""
        <div class="main-header">
            <h1 style="margin: 0; font-size: 2.5rem;">⚡ FERC EQR Analytics Dashboard</h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.2rem;">
                Interactive Data Exploration & Insights Platform
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Add a quick start guide
    with st.expander("🚀 **Quick Start Guide** - Click here if you're new!", expanded=True):
        st.markdown("""
        ### Welcome to FERC EQR Analytics! Here's how to explore your data:
        
        1. **📊 Select a Table** - Choose from available tables in the left sidebar
        2. **📋 View Data** - See your data in the "Data Preview" tab (opens by default)
        3. **📈 Create Charts** - Click the "Visualizations" tab for interactive charts
        4. **🔍 Get Insights** - Use "Advanced Analytics" tab for statistical analysis
        5. **⚙️ Customize** - Adjust settings in the left sidebar (sample size, export options)
        6. **📥 Export** - Download your data as CSV or JSON
        
        **💡 Tip**: Hover over any element for helpful tooltips!
        """)
    
    # Quick stats banner
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 1rem; border-radius: 8px; margin: 1rem 0; text-align: center;">
        <h4 style="color: white; margin: 0;">
            👈 Start by selecting a table from the sidebar, then explore the tabs below! 👇
        </h4>
    </div>
    """, unsafe_allow_html=True)


def create_sidebar_filters(tables, libsql_adapter):
    """Create minimal sidebar with table selection and basic settings."""
    # Simple header
    st.sidebar.markdown("### ⚙️ Settings")
    
    # Preview rows setting
    st.sidebar.markdown("**Preview Rows**")
    preview_limit = st.sidebar.slider(
        "Number of rows to display",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Maximum number of rows to load for preview"
    )
    
    st.sidebar.markdown("---")
    
    # Table Selection
    st.sidebar.markdown("### �️ Table Explorer")
    st.sidebar.markdown("**Select a table to analyze**")
    
    # Enhanced table selection with descriptions
    table_descriptions = {
        "contacts": "👥 Contact information and details",
        "contract_products": "📦 Products and services in contracts", 
        "contracts": "📄 Contract agreements and terms",
        "organizations": "🏢 Company and organization data",
        "transactions": "💰 Financial transactions and payments"
    }
    
    # Create a selectbox for table selection
    selected_table = st.sidebar.selectbox(
        "Choose a table:",
        options=tables,
        key="selected_table",
        help="Select a table to explore its data",
        format_func=lambda x: f"{x.replace('_', ' ').title()}"
    )
    
    if selected_table:
        description = table_descriptions.get(selected_table, "📊 Database table with structured data")
        st.sidebar.info(f"📋 {description}")
        
        # Show table info
        try:
            row_count = libsql_table_row_count(libsql_adapter, selected_table)
            st.sidebar.metric("Total Records", f"{row_count:,}")
        except Exception:
            st.sidebar.warning("⚠️ Could not load table info")
    
    # Fixed settings (no UI controls needed)
    show_advanced = True  # Always enabled
    show_sql = False      # Always disabled
    auto_refresh = False  # Always disabled
    refresh_interval = 60
    sample_percent = 0    # Always show all data
    export_format = "CSV" # Default format
    
    return selected_table, preview_limit, sample_percent, show_advanced, show_sql, auto_refresh, refresh_interval, export_format


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


def create_table_selector(tables, libsql_adapter, preview_limit, sample_percent, show_sql, export_format):
    """Create an interactive table selector with rich information."""
    st.markdown("### 🗂️ Table Explorer")
    st.markdown("**Step 1:** Choose a table to start exploring your data")
    
    # Enhanced table selection with descriptions
    table_descriptions = {
        "contacts": "👥 Contact information and details",
        "contract_products": "📦 Products and services in contracts", 
        "contracts": "📄 Contract agreements and terms",
        "organizations": "🏢 Company and organization data",
        "transactions": "💰 Financial transactions and payments"
    }
    
    # Create a more visual table selector
    selected_table = None
    
    st.markdown("**Available Tables:**")
    for i, table in enumerate(tables):
        description = table_descriptions.get(table, "📊 Database table with structured data")
        
        # Create clickable cards for each table
        if st.button(
            f"📋 **{table.title()}** - {description}", 
            key=f"table_btn_{i}",
            use_container_width=True,
            help=f"Click to explore the {table} table"
        ):
            st.session_state.selected_table = table
    
    # Get the selected table
    selected_table = st.session_state.get('selected_table', tables[0] if tables else None)
    
    if selected_table:
        # Show selection confirmation
        st.success(f"✅ **Selected Table:** {selected_table}")
        
        # Get table info with loading indicator
        with st.spinner("Loading table information..."):
            try:
                row_count = libsql_table_row_count(libsql_adapter, selected_table)
                
                # Beautiful table info card with animation-like styling
                st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 2rem;
                        border-radius: 15px;
                        color: white;
                        margin: 1rem 0;
                        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                        border: 1px solid rgba(255,255,255,0.2);
                        backdrop-filter: blur(10px);
                    ">
                        <div style="text-align: center;">
                            <h3 style="margin: 0 0 1rem 0; font-size: 1.8rem;">📊 {selected_table.title()} Table</h3>
                            <div style="display: flex; justify-content: space-around; align-items: center;">
                                <div style="text-align: center;">
                                    <p style="margin: 0; font-size: 2.5rem; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{row_count:,}</p>
                                    <p style="margin: 0; opacity: 0.9; font-size: 1.1rem;">Total Records</p>
                                </div>
                                <div style="text-align: center;">
                                    <p style="margin: 0; font-size: 2rem; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{min(row_count, preview_limit):,}</p>
                                    <p style="margin: 0; opacity: 0.9; font-size: 1.1rem;">Preview Rows</p>
                                </div>
                            </div>
                            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px;">
                                <p style="margin: 0; font-size: 1.1rem; opacity: 0.95;">
                                    💡 <strong>Next Step:</strong> Explore your data using the tabs on the right →
                                </p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error loading table info: {e}")
        
        # Interactive action buttons with better styling
        st.markdown("### ⚡ Quick Actions")
        st.markdown("**Step 2:** Use these tools to work with your data")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 **Refresh Data**", use_container_width=True, help="Reload the latest data from the database"):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            if st.button("📥 **Export Data**", use_container_width=True, help="Download your data in CSV or JSON format"):
                with st.spinner("Preparing download..."):
                    try:
                        df = libsql_read_table(libsql_adapter, selected_table, limit=None)
                        if export_format == "CSV":
                            csv = df.to_csv(index=False)
                            st.download_button(
                                "📥 Download CSV File",
                                csv,
                                file_name=f"{selected_table}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True,
                                help="Click to download the complete dataset as CSV"
                            )
                        elif export_format == "JSON":
                            json_str = df.to_json(indent=2, orient="records")
                            st.download_button(
                                "📥 Download JSON File",
                                json_str,
                                file_name=f"{selected_table}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True,
                                help="Click to download the complete dataset as JSON"
                            )
                        st.success("✅ Download ready! Click the button above.")
                    except Exception as e:
                        st.error(f"❌ Export failed: {e}")
    else:
        st.info("👈 Select a table from the options above to begin exploring!")
        
    return selected_table


def create_data_explorer(libsql_adapter, selected_table, preview_limit, sample_percent, show_advanced, show_sql):
    """Create the main data exploration interface with enhanced discoverability."""
    # Enhanced header with visual guidance
    st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            margin: 1rem 0 2rem 0;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        ">
            <h2 style="margin: 0 0 1rem 0; font-size: 2rem;">🔍 Data Explorer: {selected_table.title()}</h2>
            <p style="margin: 0; font-size: 1.1rem; opacity: 0.9;">
                <strong>Step 3:</strong> Navigate through the tabs below to explore different aspects of your data
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
                <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px;">
                    <span>📋 Raw Data</span>
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px;">
                    <span>📊 Charts</span>
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px;">
                    <span>🔍 Analytics</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Tab descriptions for better discoverability
    st.markdown("""
    **💡 What's in each tab:**
    - **📋 Data Preview**: View raw data, column details, and basic statistics
    - **📊 Visualizations**: Interactive charts and graphs to understand patterns
    - **🔍 Advanced Analytics**: Deep insights, correlations, and data quality metrics
    """)
    
    try:
        # Load data with enhanced loading feedback
        with st.spinner(f"🔄 Loading {selected_table} data... This may take a moment for large tables."):
            df = libsql_read_table(libsql_adapter, selected_table, limit=preview_limit)
            
            # Apply sampling if specified
            if sample_percent > 0 and sample_percent < 100:
                sample_size = int(len(df) * sample_percent / 100)
                df = df.sample(n=min(sample_size, len(df)), random_state=42)
                st.success(f"✅ Loaded {sample_percent}% sample ({len(df):,} rows from {selected_table})")
            elif sample_percent == 0:
                st.success(f"✅ Loaded all preview data ({len(df):,} rows from {selected_table})")
            
            if df.empty:
                st.warning("⚠️ No data found in this table.")
                st.markdown("""
                **Possible reasons:**
                - The table is empty
                - Connection issues with the database
                - Permissions may not allow data access
                """)
                return
        
        # Create enhanced tabs with descriptions
        tab1, tab2, tab3 = st.tabs([
            "📋 Data Preview (Raw Data)", 
            "📊 Visualizations (Charts)", 
            "🔍 Advanced Analytics (Insights)"
        ])
        
        with tab1:
            st.markdown("### 📋 Raw Data View")
            st.markdown("**Explore your data table with sortable columns and detailed information**")
            create_data_preview_tab(df, selected_table, show_sql)
        
        with tab2:
            st.markdown("### 📊 Interactive Visualizations")
            st.markdown("**Create charts and graphs to understand data patterns and trends**")
            create_visualizations_tab(df, selected_table)
        
        with tab3:
            st.markdown("### 🔍 Advanced Data Analytics")
            if show_advanced:
                st.markdown("**Deep statistical analysis and data quality insights**")
                create_advanced_analytics_tab(df, selected_table)
            else:
                st.markdown("**Statistical analysis and data quality assessment**")
                st.info("🔍 Advanced Analytics are currently disabled. Enable them in the sidebar settings to see detailed insights!")
                st.markdown("**What you'll see when enabled:**")
                st.markdown("""
                - 📈 Comprehensive statistical summaries
                - 🕳️ Missing value analysis and patterns
                - 🎯 Data quality metrics and recommendations
                - 🔗 Column correlation analysis
                - 📊 Distribution analysis and outlier detection
                """)
        
    except Exception as e:
        st.error(f"❌ Error loading data from {selected_table}: {e}")
        st.info("💡 **Troubleshooting Tips:**")
        st.markdown("""
        - Check your internet connection
        - Verify the table name is correct
        - Try refreshing the data using the button in Table Explorer
        - Contact support if the issue persists
        """)


def create_data_preview_tab(df, table_name, show_sql):
    """Create the data preview tab with enhanced table display."""
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", f"{len(df.columns):,}")
    with col3:
        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col4:
        missing_cells = df.isnull().sum().sum()
        st.metric("Missing Values", f"{missing_cells:,}")
    
    st.markdown("---")
    
    # Column information
    with st.expander("📋 Column Information", expanded=False):
        col_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            col_info.append({
                "Column": col,
                "Type": dtype,
                "Non-Null": f"{len(df) - null_count:,}",
                "Null": f"{null_count:,}",
                "Unique": f"{unique_count:,}"
            })
        
        col_df = pd.DataFrame(col_info)
        st.dataframe(col_df, use_container_width=True, hide_index=True)
    
    # Show SQL query if requested
    if show_sql:
        st.code(f"SELECT * FROM {table_name} LIMIT {len(df)}", language="sql")
    
    # Enhanced data display
    st.markdown("#### 📊 Data Table")
    st.dataframe(
        df,
        use_container_width=True,
        height=400,
        hide_index=True
    )


def create_visualizations_tab(df, table_name):
    """Create interactive visualizations with enhanced user guidance."""
    
    if df.empty:
        st.warning("⚠️ No data available for visualization.")
        return
    
    # Interactive guidance header
    st.markdown("""
        <div style="
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
            padding: 1rem;
            border-radius: 10px;
            color: #333;
            margin-bottom: 1.5rem;
            border-left: 5px solid #ff6b9d;
        ">
            <h4 style="margin: 0 0 0.5rem 0;">📊 Interactive Chart Builder</h4>
            <p style="margin: 0; opacity: 0.8;">
                Select columns below to create different types of visualizations. Each chart is interactive - hover for details!
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Get numeric and categorical columns with user feedback
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not numeric_cols and not categorical_cols:
        st.error("❌ No suitable columns found for visualization.")
        st.markdown("""
        **This might happen because:**
        - All columns contain complex data types
        - The data needs preprocessing
        - Table structure is not suitable for basic charts
        """)
        return
    
    # Display available column types with guidance
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        if numeric_cols:
            st.success(f"✅ **{len(numeric_cols)} Numeric columns** found")
            st.caption("Perfect for distributions, trends, and correlations")
        else:
            st.info("ℹ️ No numeric columns available")
    
    with col_info2:
        if categorical_cols:
            st.success(f"✅ **{len(categorical_cols)} Categorical columns** found")
            st.caption("Great for counts, categories, and groupings")
        else:
            st.info("ℹ️ No categorical columns available")
    
    # Create visualization options with better UI
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        if numeric_cols:
            st.markdown("### 📈 **Distribution Charts**")
            st.markdown("*Explore the spread and frequency of numeric values*")
            
            selected_numeric = st.selectbox(
                "Choose a numeric column:",
                numeric_cols,
                key="viz_numeric",
                help="Select a column to see its distribution pattern"
            )
            
            if selected_numeric:
                try:
                    with st.spinner("Creating distribution chart..."):
                        fig = px.histogram(
                            df, 
                            x=selected_numeric,
                            title=f"📊 Distribution of {selected_numeric}",
                            template="plotly_white",
                            nbins=30
                        )
                        fig.update_layout(
                            title_font_size=16,
                            margin=dict(l=0, r=0, t=50, b=0),
                            height=350,
                            showlegend=False
                        )
                        fig.update_traces(hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add insights
                        mean_val = df[selected_numeric].mean()
                        median_val = df[selected_numeric].median()
                        std_val = df[selected_numeric].std()
                        
                        insight_col1, insight_col2, insight_col3 = st.columns(3)
                        with insight_col1:
                            st.metric("Mean", f"{mean_val:.2f}")
                        with insight_col2:
                            st.metric("Median", f"{median_val:.2f}")
                        with insight_col3:
                            st.metric("Std Dev", f"{std_val:.2f}")
                            
                except Exception as e:
                    st.error(f"❌ Error creating histogram: {e}")
                    st.info("💡 Try selecting a different column or check if the data contains valid numbers.")
    
    with viz_col2:
        if categorical_cols:
            st.markdown("### 📊 **Category Analysis**")
            st.markdown("*Analyze frequency and distribution of categories*")
            
            selected_cat = st.selectbox(
                "Choose a categorical column:",
                categorical_cols,
                key="viz_cat",
                help="Select a column to see category counts and distributions"
            )
            
            if selected_cat:
                try:
                    with st.spinner("Creating category chart..."):
                        # Get top categories to avoid overcrowded plots
                        value_counts = df[selected_cat].value_counts().head(15)
                        
                        if len(value_counts) > 0:
                            fig = px.bar(
                                x=value_counts.index,
                                y=value_counts.values,
                                title=f"📈 Top Categories in {selected_cat}",
                                template="plotly_white"
                            )
                            fig.update_layout(
                                title_font_size=16,
                                margin=dict(l=0, r=0, t=50, b=0),
                                height=350,
                                xaxis_tickangle=-45,
                                showlegend=False,
                                xaxis_title=selected_cat,
                                yaxis_title="Count"
                            )
                            fig.update_traces(hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add category insights
                            total_categories = df[selected_cat].nunique()
                            most_common = value_counts.index[0] if len(value_counts) > 0 else "N/A"
                            
                            insight_col1, insight_col2 = st.columns(2)
                            with insight_col1:
                                st.metric("Total Categories", total_categories)
                            with insight_col2:
                                st.metric("Most Common", most_common)
                        else:
                            st.info("ℹ️ No data to visualize in selected column.")
                except Exception as e:
                    st.error(f"❌ Error creating bar chart: {e}")
                    st.info("💡 Try selecting a different column or check for data quality issues.")
    
    # Enhanced correlation heatmap section
    if len(numeric_cols) > 1:
        st.markdown("---")
        st.markdown("### 🔥 **Correlation Analysis**")
        st.markdown("*Discover relationships between numeric variables*")
        
        # Allow users to select which columns to correlate
        if len(numeric_cols) > 5:
            st.markdown("**Select columns for correlation analysis:** *(Choose 2-8 columns for best results)*")
            selected_for_corr = st.multiselect(
                "Columns to analyze:",
                numeric_cols,
                default=numeric_cols[:5],
                help="Select columns to include in correlation analysis. Too many columns can make the heatmap hard to read."
            )
        else:
            selected_for_corr = numeric_cols
        
        if len(selected_for_corr) >= 2:
            try:
                with st.spinner("Computing correlations..."):
                    corr_matrix = df[selected_for_corr].corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        title="🔥 Correlation Heatmap - Discover Data Relationships",
                        template="plotly_white",
                        color_continuous_scale="RdBu_r",
                        aspect="auto",
                        text_auto=True
                    )
                    fig.update_layout(
                        title_font_size=16,
                        margin=dict(l=0, r=0, t=60, b=0),
                        height=max(400, len(selected_for_corr) * 40)
                    )
                    fig.update_traces(hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.2f}<extra></extra>')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add correlation insights
                    st.markdown("**💡 Correlation Insights:**")
                    # Find strongest positive and negative correlations
                    corr_values = corr_matrix.values
                    mask = np.triu(np.ones_like(corr_values, dtype=bool), k=1)
                    corr_values[~mask] = np.nan
                    
                    max_corr_idx = np.nanargmax(corr_values)
                    min_corr_idx = np.nanargmin(corr_values)
                    
                    max_row, max_col = np.unravel_index(max_corr_idx, corr_values.shape)
                    min_row, min_col = np.unravel_index(min_corr_idx, corr_values.shape)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"🔗 **Strongest Positive:** {selected_for_corr[max_row]} ↔ {selected_for_corr[max_col]} ({corr_values[max_row, max_col]:.3f})")
                    with col2:
                        st.info(f"🔗 **Strongest Negative:** {selected_for_corr[min_row]} ↔ {selected_for_corr[min_col]} ({corr_values[min_row, min_col]:.3f})")
                        
            except Exception as e:
                st.error(f"❌ Error creating correlation heatmap: {e}")
                st.info("💡 This might happen with non-numeric data or insufficient data points.")
        else:
            st.info("📊 Select at least 2 numeric columns to see correlations.")
    else:
        st.info("📊 Need at least 2 numeric columns for correlation analysis.")
    
    # Pro tips section
    with st.expander("💡 **Pro Tips for Better Visualizations**", expanded=False):
        st.markdown("""
        **Making the Most of Your Charts:**
        
        📊 **Distribution Charts:**
        - Look for patterns: normal, skewed, or multi-modal distributions
        - Identify outliers that might need attention
        - Compare mean vs median to understand skewness
        
        📈 **Category Charts:**
        - Identify dominant categories in your data
        - Look for unexpected patterns or rare categories
        - Consider data quality issues with too many unique values
        
        🔥 **Correlation Heatmaps:**
        - Values close to +1: strong positive relationship
        - Values close to -1: strong negative relationship  
        - Values close to 0: little to no linear relationship
        - Use insights to guide further analysis
        
        **Interaction Tips:**
        - Hover over chart elements for detailed information
        - Use chart controls to zoom and pan
        - Click legend items to show/hide data series
        """)
        
    # Data quality reminders
    st.markdown("---")
    st.markdown("**🔍 Chart Quality Notes:**")
    st.caption(f"Displaying visualizations for {len(df)} rows from {table_name}. Charts are interactive - hover and click to explore!")
    
    if len(df) > 10000:
        st.info("💡 **Large Dataset Detected:** Consider using sampling in the sidebar for faster chart rendering.")
    
    if len(df) < 10:
        st.warning("⚠️ **Small Dataset:** Results may not be statistically meaningful with fewer than 10 rows.")


def create_advanced_analytics_tab(df, table_name):
    """Create advanced analytics and insights."""
    
    st.markdown("#### 🧮 Statistical Summary")
    
    # Get numeric columns for statistical analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        # Statistical summary
        desc_stats = df[numeric_cols].describe()
        st.dataframe(desc_stats, use_container_width=True)
        
        # Missing value analysis
        st.markdown("#### 🔍 Missing Value Analysis")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': missing_percent.values
        }).sort_values('Missing %', ascending=False)
        
        # Only show columns with missing data
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
            
            # Visualization of missing data
            try:
                fig = px.bar(
                    missing_df.head(10),
                    x='Column',
                    y='Missing %',
                    title='Missing Data by Column (Top 10)',
                    template="plotly_white"
                )
                fig.update_layout(
                    title_font_size=16,
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=300,
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating missing data visualization: {e}")
        else:
            st.success("✅ No missing values found!")
    
    # Data quality metrics
    st.markdown("#### ⚡ Data Quality Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        completeness = ((df.size - df.isnull().sum().sum()) / df.size) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
    
    with col2:
        duplicate_rows = df.duplicated().sum()
        st.metric("Duplicate Rows", f"{duplicate_rows:,}")
    
    with col3:
        unique_ratio = (df.nunique().sum() / df.size) * 100
        st.metric("Uniqueness Ratio", f"{unique_ratio:.1f}%")


# === MAIN APPLICATION ===

def main():
    # Create beautiful dashboard header
    create_dashboard_header()
    
    # Get connection info automatically
    db_input = get_connection_info()
    
    # Check if we have a database connection
    if not db_input:
        st.error("❌ No database connection configured.")
        st.markdown("""
        **Please set up your database connection:**
        
        **For Turso (Cloud Database):**
        - Set `TURSO_DATABASE_URL` (e.g., `libsql://your-database-name.turso.io`)
        - Set `TURSO_AUTH_TOKEN` (your read token)
        
        **For other databases:**
        - Set `DATABASE_URL` (e.g., `postgresql://...`, `mysql://...`, etc.)
        
        **For local development:**
        - Place your SQLite database at `ferc-eqr-scraper/ferc_data.db`
        
        💡 In Streamlit Cloud, add these as secrets in your app settings.
        """)
        st.stop()
    
    # Create database connection with progress feedback
    libsql_adapter = None
    try:
        with st.spinner("🔌 Connecting to database..."):
            auth_token = os.environ.get("TURSO_AUTH_TOKEN")
            libsql_adapter = get_libsql_client(db_input['url'], auth_token=auth_token)
            if not isinstance(libsql_adapter, dict) or "type" not in libsql_adapter:
                st.error(f"❌ Database adapter error: {libsql_adapter!r}")
                st.stop()
        
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        st.stop()

    # Get tables with progress feedback
    try:
        with st.spinner("📋 Loading database schema..."):
            tables = libsql_list_tables(libsql_adapter)
        
    except Exception as e:
        st.error(f"❌ Failed to retrieve database schema: {e}")
        st.stop()

    if not tables:
        st.warning("⚠️ No tables found in the database.")
        st.stop()

    # Get sidebar controls (now includes table selection)
    selected_table, preview_limit, sample_percent, show_advanced, show_sql, auto_refresh, refresh_interval, export_format = create_sidebar_filters(tables, libsql_adapter)
    
    # Full-width data explorer (no columns layout)
    if selected_table:
        # Show data explorer for the selected table using full width
        create_data_explorer(libsql_adapter, selected_table, preview_limit, sample_percent, show_advanced, show_sql)
    else:
        # Show database overview and welcome state when no table is selected
        create_database_overview(libsql_adapter, tables)
        
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                color: white;
                margin: 2rem 0;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            ">
                <h2 style="margin: 0 0 1rem 0; font-size: 2rem;">🚀 Welcome to FERC EQR Analytics!</h2>
                <p style="margin: 0 0 1.5rem 0; font-size: 1.1rem; opacity: 0.9;">
                    Select a table from the sidebar to start exploring your data
                </p>
                <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1.5rem;">
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                        <div>Interactive Charts</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
                        <div>Advanced Analytics</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">📥</div>
                        <div>Data Export</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()