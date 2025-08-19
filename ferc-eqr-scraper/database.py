"""Database component for FERC EQR data storage."""

import time
from typing import Dict, List, Optional, Any
import pandas as pd
import logging
from sqlalchemy import create_engine, text, inspect, MetaData, Table
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.pool import QueuePool

from settings import config
from utils import format_bytes


class FERCDatabase:
    """Database manager for FERC EQR data storage."""
    
    def __init__(self, database_uri: Optional[str] = None):
        """Initialize database connection.
        
        Args:
            database_uri: Database connection string (defaults to config)
        """
        self.logger = logging.getLogger("ferc_scraper.database")
        self.database_uri = database_uri or config.DATABASE_URI
        self.engine: Optional[Engine] = None
        self._connection_pool_size = 5
        self._max_overflow = 10
        
        # Initialize connection
        self._create_engine()
    
    def _create_engine(self) -> None:
        """Create SQLAlchemy engine with appropriate configuration."""
        try:
            engine_kwargs = {
                'echo': False,  # Set to True for SQL debugging
                'pool_pre_ping': True,  # Verify connections before use
            }
            
            # Configure for different database types
            if config.is_turso():
                self.logger.info("Configuring for Turso/LibSQL database")
                # Turso-specific optimizations
                engine_kwargs.update({
                    'poolclass': QueuePool,
                    'pool_size': 3,  # Smaller pool for serverless
                    'max_overflow': 5,
                    'pool_recycle': 300,  # Recycle connections more frequently
                    'connect_args': {
                        'check_same_thread': False,  # Allow multi-threading
                        'timeout': 30,
                    }
                })
                
            elif config.is_postgresql():
                self.logger.info("Configuring for PostgreSQL database")
                engine_kwargs.update({
                    'poolclass': QueuePool,
                    'pool_size': self._connection_pool_size,
                    'max_overflow': self._max_overflow,
                    'pool_recycle': 3600,  # Recycle connections after 1 hour
                })
                
                # Try to import psycopg2
                try:
                    import psycopg2
                    self.logger.info("Using psycopg2 driver for PostgreSQL")
                except ImportError:
                    self.logger.warning("psycopg2 not available, falling back to default driver")
                    
            else:
                # Local SQLite
                engine_kwargs.update({
                    'connect_args': {'check_same_thread': False}
                })
            
            self.engine = create_engine(self.database_uri, **engine_kwargs)
            
            # Test connection
            self._test_connection()
            
            db_type = config.get_database_type()
            self.logger.info(f"Database engine created: {db_type}")
            
            if config.is_turso():
                self.logger.info("Connected to Turso database successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create database engine: {e}")
            raise
    
    def _test_connection(self) -> None:
        """Test database connection."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.logger.info("Database connection test successful")
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            raise
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about the database connection.
        
        Returns:
            Dictionary with connection information
        """
        info = {
            'database_type': config.get_database_type(),
            'database_uri': self.database_uri.split('@')[-1] if '@' in self.database_uri else self.database_uri,
            'engine_created': self.engine is not None,
            'pool_size': getattr(self.engine.pool, 'size', None) if self.engine else None,
            'pool_checked_out': getattr(self.engine.pool, 'checkedout', None) if self.engine else None,
        }
        
        return info
    
    def is_sqlite(self) -> bool:
        """Check if the database is SQLite.
        
        Returns:
            True if database is SQLite, False otherwise
        """
        return config.is_sqlite()
    
    def is_turso(self) -> bool:
        """Check if the database is Turso.
        
        Returns:
            True if database is Turso, False otherwise
        """
        return config.is_turso()
    
    def is_postgresql(self) -> bool:
        """Check if the database is PostgreSQL.
        
        Returns:
            True if database is PostgreSQL, False otherwise
        """
        return config.is_postgresql()
    
    def execute_with_retry(self, operation, max_retries: int = None) -> Any:
        """Execute database operation with retry logic.
        
        Args:
            operation: Function to execute
            max_retries: Maximum number of retries (defaults to config.MAX_RETRIES)
            
        Returns:
            Result of the operation
            
        Raises:
            SQLAlchemyError: If operation fails after all retries
        """
        if max_retries is None:
            max_retries = config.MAX_RETRIES
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return operation()
            except OperationalError as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(f"Database operation failed (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Database operation failed after {max_retries + 1} attempts")
                    raise
            except Exception as e:
                # Don't retry for non-operational errors
                self.logger.error(f"Database operation failed with non-retryable error: {e}")
                raise
        
        # This should never be reached, but just in case
        raise last_exception
    
    def get_existing_tables(self) -> List[str]:
        """Get list of existing tables in the database.
        
        Returns:
            List of table names
        """
        def _get_tables():
            inspector = inspect(self.engine)
            return inspector.get_table_names()
        
        try:
            tables = self.execute_with_retry(_get_tables)
            self.logger.info(f"Found {len(tables)} existing tables")
            return tables
        except Exception as e:
            self.logger.error(f"Error getting table list: {e}")
            return []
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if table exists, False otherwise
        """
        def _check_table():
            inspector = inspect(self.engine)
            return inspector.has_table(table_name)
        
        try:
            return self.execute_with_retry(_check_table)
        except Exception as e:
            self.logger.error(f"Error checking if table {table_name} exists: {e}")
            return False
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table information
        """
        def _get_info():
            with self.engine.connect() as conn:
                # Get row count
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                row_count = result.scalar()
                
                # Get column info
                inspector = inspect(self.engine)
                columns = inspector.get_columns(table_name)
                
                return {
                    'row_count': row_count,
                    'column_count': len(columns),
                    'columns': [col['name'] for col in columns]
                }
        
        try:
            return self.execute_with_retry(_get_info)
        except Exception as e:
            self.logger.error(f"Error getting info for table {table_name}: {e}")
            return {'row_count': 0, 'column_count': 0, 'columns': []}
    
    def get_existing_data_info(self) -> Dict[str, Any]:
        """Get information about existing data in the database.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            tables = self.get_existing_tables()
            
            info = {
                'total_tables': len(tables),
                'tables': {}
            }
            
            for table_name in tables:
                table_info = self.get_table_info(table_name)
                info['tables'][table_name] = table_info
            
            total_rows = sum(table_info['row_count'] for table_info in info['tables'].values())
            info['total_rows'] = total_rows
            
            self.logger.info(f"Database contains {len(tables)} tables with {total_rows} total rows")
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting database info: {e}")
            return {'total_tables': 0, 'tables': {}, 'total_rows': 0}
    
    def create_production_schema(self) -> None:
        """Create production-ready database schema with proper indexes and constraints.
        
        Optimized for Turso/LibSQL with efficient indexes and foreign key constraints.
        """
        try:
            with self.engine.connect() as conn:
                trans = conn.begin()
                try:
                    # Enable foreign key constraints (important for SQLite/Turso)
                    conn.execute(text("PRAGMA foreign_keys = ON"))
                    
                    # Organizations table with proper constraints
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS organizations (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            organization_uid TEXT NOT NULL,
                            cid TEXT,
                            company_name TEXT,
                            is_filer BOOLEAN DEFAULT FALSE,
                            is_buyer BOOLEAN DEFAULT FALSE,
                            is_seller BOOLEAN DEFAULT FALSE,
                            transactions_reported_to_index_publisher BOOLEAN DEFAULT FALSE,
                            filing_uid TEXT NOT NULL,
                            year INTEGER NOT NULL,
                            quarter INTEGER NOT NULL,
                            period_type TEXT,
                            filing_type TEXT,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(organization_uid, year, quarter)
                        )
                    """))
                    
                    # Contacts table with foreign key to organizations
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS contacts (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            contact_uid TEXT NOT NULL,
                            organization_uid TEXT NOT NULL,
                            first_name TEXT,
                            last_name TEXT,
                            display_name TEXT,
                            title TEXT,
                            phone TEXT,
                            email TEXT,
                            is_filer_contact BOOLEAN DEFAULT FALSE,
                            is_buyer_contact BOOLEAN DEFAULT FALSE,
                            is_seller_contact BOOLEAN DEFAULT FALSE,
                            street1 TEXT,
                            street2 TEXT,
                            street3 TEXT,
                            city TEXT,
                            state TEXT,
                            zip TEXT,
                            country TEXT,
                            filing_uid TEXT NOT NULL,
                            year INTEGER NOT NULL,
                            quarter INTEGER NOT NULL,
                            period_type TEXT,
                            filing_type TEXT,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(contact_uid, organization_uid, year, quarter),
                            FOREIGN KEY (organization_uid, year, quarter) 
                                REFERENCES organizations (organization_uid, year, quarter)
                        )
                    """))
                    
                    # Contracts table
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS contracts (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            contract_uid TEXT NOT NULL,
                            seller_uid TEXT,
                            buyer_uid TEXT,
                            ferc_tariff_reference TEXT,
                            contract_service_agreement TEXT,
                            is_affiliate BOOLEAN DEFAULT FALSE,
                            execution_date DATE,
                            commencement_date DATE,
                            termination_date DATE,
                            extension_provision_description TEXT,
                            filing_type_contract TEXT,
                            filing_uid TEXT NOT NULL,
                            year INTEGER NOT NULL,
                            quarter INTEGER NOT NULL,
                            period_type TEXT,
                            filing_type TEXT,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(contract_uid, year, quarter),
                            FOREIGN KEY (seller_uid, year, quarter) 
                                REFERENCES organizations (organization_uid, year, quarter),
                            FOREIGN KEY (buyer_uid, year, quarter) 
                                REFERENCES organizations (organization_uid, year, quarter)
                        )
                    """))
                    
                    # Contract products table
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS contract_products (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            product_uid TEXT NOT NULL,
                            contract_uid TEXT NOT NULL,
                            product_type TEXT,
                            product_name TEXT,
                            product_class TEXT,
                            term TEXT,
                            increment TEXT,
                            increment_peaking TEXT,
                            quantity DECIMAL(15,4),
                            units TEXT,
                            podsl TEXT,
                            begin_date DATE,
                            end_date DATE,
                            rate_description TEXT,
                            rate_units TEXT,
                            filing_type_product TEXT,
                            filing_uid TEXT NOT NULL,
                            year INTEGER NOT NULL,
                            quarter INTEGER NOT NULL,
                            period_type TEXT,
                            filing_type TEXT,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(product_uid, contract_uid, year, quarter),
                            FOREIGN KEY (contract_uid, year, quarter) 
                                REFERENCES contracts (contract_uid, year, quarter)
                        )
                    """))
                    
                    # Transactions table
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS transactions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            transaction_uid TEXT NOT NULL,
                            contract_uid TEXT NOT NULL,
                            transaction_group_ref TEXT,
                            begin_date DATE,
                            end_date DATE,
                            time_zone TEXT,
                            trade_date DATE,
                            podba TEXT,
                            podsl TEXT,
                            transaction_class TEXT,
                            term TEXT,
                            increment TEXT,
                            increment_peaking TEXT,
                            product_name TEXT,
                            quantity DECIMAL(15,4),
                            standardized_quantity DECIMAL(15,4),
                            price DECIMAL(15,4),
                            standardized_price DECIMAL(15,4),
                            rate_units TEXT,
                            rate_type TEXT,
                            total_transmission_charge DECIMAL(15,4),
                            transaction_charge DECIMAL(15,4),
                            filing_type_transaction TEXT,
                            filing_uid TEXT NOT NULL,
                            year INTEGER NOT NULL,
                            quarter INTEGER NOT NULL,
                            period_type TEXT,
                            filing_type TEXT,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(transaction_uid, contract_uid, year, quarter),
                            FOREIGN KEY (contract_uid, year, quarter) 
                                REFERENCES contracts (contract_uid, year, quarter)
                        )
                    """))
                    
                    trans.commit()
                    self.logger.info("Production schema created successfully")
                    
                except Exception as e:
                    trans.rollback()
                    raise e
                    
        except Exception as e:
            self.logger.error(f"Failed to create production schema: {e}")
            raise

    def create_production_indexes(self) -> None:
        """Create production-ready indexes for optimal query performance."""
        try:
            with self.engine.connect() as conn:
                trans = conn.begin()
                try:
                    indexes = [
                        # Organizations indexes
                        "CREATE INDEX IF NOT EXISTS idx_organizations_uid ON organizations (organization_uid)",
                        "CREATE INDEX IF NOT EXISTS idx_organizations_year_quarter ON organizations (year, quarter)",
                        "CREATE INDEX IF NOT EXISTS idx_organizations_company_name ON organizations (company_name)",
                        "CREATE INDEX IF NOT EXISTS idx_organizations_is_seller ON organizations (is_seller) WHERE is_seller = 1",
                        "CREATE INDEX IF NOT EXISTS idx_organizations_is_buyer ON organizations (is_buyer) WHERE is_buyer = 1",
                        
                        # Contacts indexes
                        "CREATE INDEX IF NOT EXISTS idx_contacts_org_uid ON contacts (organization_uid)",
                        "CREATE INDEX IF NOT EXISTS idx_contacts_email ON contacts (email) WHERE email IS NOT NULL",
                        "CREATE INDEX IF NOT EXISTS idx_contacts_year_quarter ON contacts (year, quarter)",
                        
                        # Contracts indexes
                        "CREATE INDEX IF NOT EXISTS idx_contracts_uid ON contracts (contract_uid)",
                        "CREATE INDEX IF NOT EXISTS idx_contracts_seller ON contracts (seller_uid)",
                        "CREATE INDEX IF NOT EXISTS idx_contracts_buyer ON contracts (buyer_uid)",
                        "CREATE INDEX IF NOT EXISTS idx_contracts_year_quarter ON contracts (year, quarter)",
                        "CREATE INDEX IF NOT EXISTS idx_contracts_execution_date ON contracts (execution_date)",
                        "CREATE INDEX IF NOT EXISTS idx_contracts_is_affiliate ON contracts (is_affiliate) WHERE is_affiliate = 1",
                        
                        # Contract products indexes
                        "CREATE INDEX IF NOT EXISTS idx_products_contract_uid ON contract_products (contract_uid)",
                        "CREATE INDEX IF NOT EXISTS idx_products_type ON contract_products (product_type)",
                        "CREATE INDEX IF NOT EXISTS idx_products_year_quarter ON contract_products (year, quarter)",
                        "CREATE INDEX IF NOT EXISTS idx_products_begin_date ON contract_products (begin_date)",
                        
                        # Transactions indexes
                        "CREATE INDEX IF NOT EXISTS idx_transactions_contract_uid ON transactions (contract_uid)",
                        "CREATE INDEX IF NOT EXISTS idx_transactions_trade_date ON transactions (trade_date)",
                        "CREATE INDEX IF NOT EXISTS idx_transactions_year_quarter ON transactions (year, quarter)",
                        "CREATE INDEX IF NOT EXISTS idx_transactions_price ON transactions (price) WHERE price IS NOT NULL",
                        "CREATE INDEX IF NOT EXISTS idx_transactions_quantity ON transactions (quantity) WHERE quantity IS NOT NULL",
                        "CREATE INDEX IF NOT EXISTS idx_transactions_product_name ON transactions (product_name)",
                        
                        # Composite indexes for common queries
                        "CREATE INDEX IF NOT EXISTS idx_organizations_composite ON organizations (year, quarter, is_seller, is_buyer)",
                        "CREATE INDEX IF NOT EXISTS idx_contracts_composite ON contracts (year, quarter, seller_uid, buyer_uid)",
                        "CREATE INDEX IF NOT EXISTS idx_transactions_composite ON transactions (year, quarter, trade_date, product_name)",
                    ]
                    
                    for index_sql in indexes:
                        conn.execute(text(index_sql))
                    
                    trans.commit()
                    self.logger.info(f"Created {len(indexes)} production indexes")
                    
                except Exception as e:
                    trans.rollback()
                    raise e
                    
        except Exception as e:
            self.logger.error(f"Failed to create production indexes: {e}")
            raise

    def optimize_for_turso(self) -> None:
        """Apply Turso/LibSQL specific optimizations."""
        try:
            with self.engine.connect() as conn:
                # Turso/LibSQL optimizations
                optimizations = [
                    "PRAGMA journal_mode = WAL",  # Write-Ahead Logging for better concurrency
                    "PRAGMA synchronous = NORMAL",  # Balance between safety and performance
                    "PRAGMA cache_size = 10000",  # Increase cache size for better performance
                    "PRAGMA temp_store = MEMORY",  # Store temporary data in memory
                    "PRAGMA mmap_size = 268435456",  # Use memory-mapped I/O (256MB)
                    "PRAGMA foreign_keys = ON",  # Enable foreign key constraints
                ]
                
                for pragma in optimizations:
                    try:
                        conn.execute(text(pragma))
                    except Exception as e:
                        self.logger.warning(f"Optimization '{pragma}' failed: {e}")
                
                self.logger.info("Applied Turso/LibSQL optimizations")
                
        except Exception as e:
            self.logger.error(f"Failed to apply Turso optimizations: {e}")

    def initialize_production_schema(self) -> None:
        """Initialize complete production-ready schema with all optimizations."""
        try:
            # Test connection
            self._test_connection()
            
            # Apply Turso optimizations first
            self.optimize_for_turso()
            
            # Create production schema
            self.create_production_schema()
            
            # Create all indexes
            self.create_production_indexes()
            
            # Get existing schema info
            existing_info = self.get_existing_data_info()
            
            self.logger.info(
                f"Production schema initialized: {existing_info['total_tables']} existing tables "
                f"with {existing_info['total_rows']} total rows"
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing production schema: {e}")
            raise
    
    def get_dataframe_schema_info(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Get schema information for a DataFrame.
        
        Args:
            df: DataFrame to analyze
            table_name: Name of the target table
            
        Returns:
            Dictionary with schema information
        """
        try:
            schema_info = {
                'table_name': table_name,
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'has_nulls': df.isnull().any().to_dict(),
                'null_counts': df.isnull().sum().to_dict()
            }
            
            # Convert numpy dtypes to strings for JSON serialization
            schema_info['dtypes'] = {col: str(dtype) for col, dtype in schema_info['dtypes'].items()}
            
            return schema_info
            
        except Exception as e:
            self.logger.error(f"Error analyzing DataFrame schema for {table_name}: {e}")
            return {}
    
    def optimize_dataframe_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for database storage.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            DataFrame with optimized data types
        """
        try:
            optimized_df = df.copy()
            
            for column in optimized_df.columns:
                col_data = optimized_df[column]
                
                # Skip if column is all null
                if col_data.isnull().all():
                    continue
                
                # Handle datetime/timestamp columns first
                if pd.api.types.is_datetime64_any_dtype(col_data):
                    # Convert Pandas Timestamps to string format for SQLite compatibility
                    if self.is_sqlite():
                        # Convert to ISO format strings, keeping NaT as None
                        optimized_df[column] = col_data.dt.strftime('%Y-%m-%d %H:%M:%S').where(col_data.notna(), None)
                    # For other databases, keep as datetime
                    else:
                        optimized_df[column] = col_data
                
                # Handle object columns that might contain Timestamps
                elif pd.api.types.is_object_dtype(col_data):
                    # Check if any values are Timestamp objects
                    sample_non_null = col_data.dropna()
                    if len(sample_non_null) > 0:
                        sample_value = sample_non_null.iloc[0]
                        if isinstance(sample_value, pd.Timestamp):
                            # Convert all Timestamp objects to strings
                            if self.is_sqlite():
                                optimized_df[column] = col_data.apply(
                                    lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if isinstance(x, pd.Timestamp) else x
                                )
                            else:
                                # For other databases, convert to proper datetime
                                optimized_df[column] = pd.to_datetime(col_data, errors='coerce')
                        else:
                            # Convert to category if many repeated values
                            unique_ratio = col_data.nunique() / len(col_data)
                            if unique_ratio < 0.5 and col_data.nunique() < 1000:
                                optimized_df[column] = col_data.astype('category')
                
                # Optimize numeric columns
                elif pd.api.types.is_numeric_dtype(col_data):
                    # Try to downcast integers
                    if pd.api.types.is_integer_dtype(col_data):
                        optimized_df[column] = pd.to_numeric(col_data, downcast='integer')
                    # Try to downcast floats
                    elif pd.api.types.is_float_dtype(col_data):
                        optimized_df[column] = pd.to_numeric(col_data, downcast='float')
            
            memory_before = df.memory_usage(deep=True).sum()
            memory_after = optimized_df.memory_usage(deep=True).sum()
            
            if memory_after < memory_before:
                reduction_pct = (1 - memory_after / memory_before) * 100
                self.logger.info(
                    f"DataFrame memory optimized: {format_bytes(memory_before)} -> "
                    f"{format_bytes(memory_after)} ({reduction_pct:.1f}% reduction)"
                )
            
            return optimized_df
            
        except Exception as e:
            self.logger.warning(f"Error optimizing DataFrame dtypes: {e}")
            return df
    
    def validate_dataframe_for_loading(self, df: pd.DataFrame, table_name: str) -> bool:
        """Validate DataFrame before loading to database.
        
        Args:
            df: DataFrame to validate
            table_name: Target table name
            
        Returns:
            True if DataFrame is valid for loading, False otherwise
        """
        try:
            if df.empty:
                self.logger.warning(f"DataFrame for table {table_name} is empty")
                return False
            
            # Check for problematic column names
            problematic_columns = []
            for col in df.columns:
                if not isinstance(col, str):
                    problematic_columns.append(col)
                elif col.strip() != col or not col:
                    problematic_columns.append(col)
            
            if problematic_columns:
                self.logger.warning(f"Table {table_name} has problematic column names: {problematic_columns}")
                return False
            
            # Check for extremely large values that might cause issues
            for col in df.select_dtypes(include=['number']).columns:
                if df[col].abs().max() > 1e15:  # Arbitrary large number threshold
                    self.logger.warning(f"Table {table_name}, column {col} has extremely large values")
            
            # Check memory usage
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            if memory_mb > 1000:  # 1GB threshold
                self.logger.warning(f"Table {table_name} uses {memory_mb:.1f} MB of memory")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating DataFrame for table {table_name}: {e}")
            return False
    
    def cleanup_failed_loads(self) -> None:
        """Clean up any failed or incomplete data loads.
        
        This method can be extended to implement specific cleanup logic
        based on your data loading patterns.
        """
        try:
            # Get all tables
            tables = self.get_existing_tables()
            
            # Look for tables with zero rows (potential failed loads)
            empty_tables = []
            for table_name in tables:
                table_info = self.get_table_info(table_name)
                if table_info['row_count'] == 0:
                    empty_tables.append(table_name)
            
            if empty_tables:
                self.logger.info(f"Found {len(empty_tables)} empty tables: {empty_tables}")
                # In a production system, you might want to drop these tables
                # For now, just log them
            
            self.logger.info("Database cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during database cleanup: {e}")
    
    def load_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> None:
        """Load a single DataFrame into the database with chunked processing.
        
        Args:
            df: DataFrame to load
            table_name: Target table name
            if_exists: How to behave if table exists ('fail', 'replace', 'append')
            
        Raises:
            Exception: If loading fails
        """
        if df.empty:
            self.logger.warning(f"Skipping empty DataFrame for table {table_name}")
            return
        
        # Validate DataFrame
        if not self.validate_dataframe_for_loading(df, table_name):
            raise ValueError(f"DataFrame validation failed for table {table_name}")
        
        # Optimize data types
        optimized_df = self.optimize_dataframe_dtypes(df)
        
        # Additional data cleaning for transactions table
        if table_name == 'transactions':
            self.logger.debug(f"Cleaning transaction data for {len(optimized_df)} rows")
            # Ensure all string columns don't have problematic values
            for col in optimized_df.columns:
                if optimized_df[col].dtype == 'object':
                    # Replace NaN with None for proper SQL handling
                    optimized_df[col] = optimized_df[col].where(optimized_df[col].notna(), None)
        
        # Get schema info
        schema_info = self.get_dataframe_schema_info(optimized_df, table_name)
        
        self.logger.info(
            f"Loading {schema_info['row_count']} rows into table '{table_name}' "
            f"({format_bytes(schema_info['memory_usage_mb'] * 1024 * 1024)})"
        )
        
        def _load_operation():
            # Use chunked loading for large DataFrames
            chunk_size = min(config.CHUNK_SIZE, len(optimized_df))
            
            try:
                with self.engine.begin() as conn:  # Transaction context
                    optimized_df.to_sql(
                        name=table_name,
                        con=conn,
                        if_exists=if_exists,
                        index=False,
                        chunksize=chunk_size,
                        method=None  # Use default method instead of 'multi' for better compatibility
                    )
            except Exception as e:
                self.logger.error(f"SQL insertion error for table {table_name}: {e}")
                self.logger.error(f"DataFrame shape: {optimized_df.shape}")
                self.logger.error(f"DataFrame columns: {list(optimized_df.columns)}")
                self.logger.error(f"DataFrame dtypes: {optimized_df.dtypes.to_dict()}")
                if len(optimized_df) > 0:
                    self.logger.error(f"Sample row: {optimized_df.iloc[0].to_dict()}")
                raise
        
        # Execute with retry logic
        self.execute_with_retry(_load_operation)
        
        # Verify the load
        final_count = self.get_table_info(table_name)['row_count']
        self.logger.info(f"Successfully loaded data to '{table_name}'. Table now has {final_count} rows.")
    
    def load_dataframes(self, dataframes: Dict[str, pd.DataFrame], if_exists: str = 'append') -> None:
        """Load multiple DataFrames into the database.
        
        Args:
            dataframes: Dictionary mapping table names to DataFrames
            if_exists: How to behave if tables exist ('fail', 'replace', 'append')
        """
        if not dataframes:
            self.logger.warning("No DataFrames to load")
            return
        
        total_rows = sum(len(df) for df in dataframes.values())
        self.logger.info(f"Loading {len(dataframes)} tables with {total_rows} total rows")
        
        successful_tables = []
        failed_tables = []
        
        for table_name, df in dataframes.items():
            try:
                self.load_dataframe(df, table_name, if_exists)
                successful_tables.append(table_name)
            except Exception as e:
                self.logger.error(f"Failed to load table {table_name}: {e}")
                failed_tables.append(table_name)
                # Continue with other tables
                continue
        
        self.logger.info(
            f"Data loading complete: {len(successful_tables)} successful, {len(failed_tables)} failed"
        )
        
        if failed_tables:
            self.logger.warning(f"Failed tables: {failed_tables}")
    
    def add_unique_constraints(self, table_name: str) -> None:
        """Add unique constraints and indexes to tables for data integrity.
        
        Args:
            table_name: Name of the table to add constraints to
        """
        try:
            with self.engine.connect() as conn:
                trans = conn.begin()
                try:
                    if table_name == 'organizations':
                        # Create unique constraint on organization_uid + year + quarter
                        conn.execute(text("""
                            CREATE UNIQUE INDEX IF NOT EXISTS idx_org_unique 
                            ON organizations (organization_uid, year, quarter)
                        """))
                        # Also create primary key if SQLite
                        if self.is_sqlite():
                            conn.execute(text("""
                                CREATE UNIQUE INDEX IF NOT EXISTS idx_org_pk 
                                ON organizations (rowid)
                            """))
                    
                    elif table_name == 'contacts':
                        conn.execute(text("""
                            CREATE UNIQUE INDEX IF NOT EXISTS idx_contact_unique 
                            ON contacts (contact_uid, organization_uid, year, quarter)
                        """))
                    
                    elif table_name == 'contracts':
                        conn.execute(text("""
                            CREATE UNIQUE INDEX IF NOT EXISTS idx_contract_unique 
                            ON contracts (contract_uid, year, quarter)
                        """))
                    
                    elif table_name == 'contract_products':
                        conn.execute(text("""
                            CREATE UNIQUE INDEX IF NOT EXISTS idx_product_unique 
                            ON contract_products (product_uid, contract_uid, year, quarter)
                        """))
                    
                    elif table_name == 'transactions':
                        conn.execute(text("""
                            CREATE UNIQUE INDEX IF NOT EXISTS idx_transaction_unique 
                            ON transactions (transaction_uid, contract_uid, year, quarter)
                        """))
                    
                    # Add general indexes for common query patterns
                    conn.execute(text(f"""
                        CREATE INDEX IF NOT EXISTS idx_{table_name}_year_quarter 
                        ON {table_name} (year, quarter)
                    """))
                    
                    trans.commit()
                    self.logger.debug(f"Added unique constraints and indexes to {table_name}")
                    
                except Exception as e:
                    trans.rollback()
                    raise e
                    
        except Exception as e:
            self.logger.warning(f"Failed to add constraints to {table_name}: {e}")

    def deduplicate_dataframe(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Remove duplicates from DataFrame based on table-specific logic.
        
        Args:
            df: DataFrame to deduplicate
            table_name: Name of the target table
            
        Returns:
            Deduplicated DataFrame
        """
        if df.empty:
            return df
        
        initial_count = len(df)
        
        # Define deduplication keys for each table
        dedup_keys = {
            'organizations': ['organization_uid', 'year', 'quarter'],
            'contacts': ['contact_uid', 'organization_uid', 'year', 'quarter'],
            'contracts': ['contract_uid', 'year', 'quarter'],
            'contract_products': ['product_uid', 'contract_uid', 'year', 'quarter'],
            'transactions': ['transaction_uid', 'contract_uid', 'year', 'quarter']
        }
        
        if table_name in dedup_keys:
            key_columns = dedup_keys[table_name]
            # Only deduplicate if all key columns exist
            available_keys = [col for col in key_columns if col in df.columns]
            if len(available_keys) == len(key_columns):
                df_dedup = df.drop_duplicates(subset=available_keys, keep='first')
                removed_count = initial_count - len(df_dedup)
                if removed_count > 0:
                    self.logger.info(f"Removed {removed_count} duplicates from {table_name}")
                return df_dedup
            else:
                self.logger.warning(f"Cannot deduplicate {table_name}: missing key columns {set(key_columns) - set(available_keys)}")
        
        return df

    def load_dataframes_batch_streaming(self, dataframes_batch: Dict[str, List[pd.DataFrame]], 
                                       if_exists: str = 'append', 
                                       batch_size: Optional[int] = None) -> None:
        """Load multiple batches of DataFrames with memory-efficient streaming.
        
        Args:
            dataframes_batch: Dictionary mapping table names to lists of DataFrames
            if_exists: How to behave if tables exist ('fail', 'replace', 'append')
            batch_size: Size of chunks for processing (defaults to config.CHUNK_SIZE)
        """
        if not dataframes_batch:
            self.logger.warning("No DataFrame batches to load")
            return
        
        if batch_size is None:
            batch_size = config.CHUNK_SIZE
        
        # Calculate totals
        total_tables = len(dataframes_batch)
        total_dataframes = sum(len(df_list) for df_list in dataframes_batch.values())
        total_rows = sum(sum(len(df) for df in df_list) for df_list in dataframes_batch.values())
        
        self.logger.info(
            f"Loading {total_tables} table types with {total_dataframes} DataFrames "
            f"and {total_rows} total rows using streaming approach (batch_size={batch_size})"
        )
        
        successful_tables = []
        failed_tables = []
        
        # Process each table separately with streaming
        for table_name, df_list in dataframes_batch.items():
            try:
                self.logger.info(f"Processing table {table_name} with {len(df_list)} DataFrames")
                
                # Process DataFrames in chunks to avoid memory issues
                total_processed = 0
                
                for i, df in enumerate(df_list, 1):
                    self.logger.debug(f"Processing DataFrame {i}/{len(df_list)} for {table_name} ({len(df)} rows)")
                    
                    if df.empty:
                        continue
                    
                    # Deduplicate the DataFrame
                    df_dedup = self.deduplicate_dataframe(df, table_name)
                    
                    # Process in chunks if DataFrame is large
                    if len(df_dedup) > batch_size:
                        self.logger.info(f"Large DataFrame for {table_name}, processing in chunks of {batch_size}")
                        
                        for chunk_start in range(0, len(df_dedup), batch_size):
                            chunk_end = min(chunk_start + batch_size, len(df_dedup))
                            chunk_df = df_dedup.iloc[chunk_start:chunk_end].copy()
                            
                            # Load the chunk
                            chunk_if_exists = if_exists if total_processed == 0 and chunk_start == 0 else 'append'
                            self.load_dataframe(chunk_df, table_name, chunk_if_exists)
                            total_processed += len(chunk_df)
                            
                            # Force garbage collection
                            import gc
                            del chunk_df
                            gc.collect()
                    else:
                        # Load small DataFrame directly
                        chunk_if_exists = if_exists if total_processed == 0 else 'append'
                        self.load_dataframe(df_dedup, table_name, chunk_if_exists)
                        total_processed += len(df_dedup)
                    
                    # Clean up memory
                    import gc
                    del df_dedup
                    gc.collect()
                
                # Add unique constraints after loading all data for this table
                self.add_unique_constraints(table_name)
                
                self.logger.info(f"Successfully loaded {total_processed} rows to table {table_name}")
                successful_tables.append(table_name)
                
            except Exception as e:
                self.logger.error(f"Failed to load table {table_name}: {e}")
                failed_tables.append(table_name)
                # Force cleanup on error
                import gc
                gc.collect()
                continue
        
        self.logger.info(
            f"Streaming batch loading complete: {len(successful_tables)} successful, {len(failed_tables)} failed"
        )
        
        if failed_tables:
            self.logger.warning(f"Failed tables: {failed_tables}")

    def load_dataframes_batch(self, dataframes_batch: Dict[str, List[pd.DataFrame]], 
                             if_exists: str = 'append') -> None:
        """Load multiple batches of DataFrames (e.g., from multiple files).
        
        Args:
            dataframes_batch: Dictionary mapping table names to lists of DataFrames
            if_exists: How to behave if tables exist ('fail', 'replace', 'append')
        """
        # Use the new streaming approach by default
        self.load_dataframes_batch_streaming(dataframes_batch, if_exists)
    
    def get_loading_progress(self, table_name: str, expected_rows: int) -> Dict[str, Any]:
        """Get progress information for data loading.
        
        Args:
            table_name: Name of the table being loaded
            expected_rows: Expected number of rows
            
        Returns:
            Dictionary with progress information
        """
        try:
            current_rows = self.get_table_info(table_name)['row_count']
            progress_pct = (current_rows / expected_rows * 100) if expected_rows > 0 else 0
            
            return {
                'table_name': table_name,
                'current_rows': current_rows,
                'expected_rows': expected_rows,
                'progress_percent': round(progress_pct, 2),
                'remaining_rows': max(0, expected_rows - current_rows)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting loading progress for {table_name}: {e}")
            return {
                'table_name': table_name,
                'current_rows': 0,
                'expected_rows': expected_rows,
                'progress_percent': 0,
                'remaining_rows': expected_rows
            }
    
    def close(self) -> None:
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            self.logger.info("Database connections closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'engine') and self.engine:
            try:
                self.engine.dispose()
            except:
                pass  # Ignore errors during cleanup