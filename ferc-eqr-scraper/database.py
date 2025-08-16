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
            
            # Configure connection pooling for PostgreSQL
            if config.is_postgresql():
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
            
            self.engine = create_engine(self.database_uri, **engine_kwargs)
            
            # Test connection
            self._test_connection()
            
            self.logger.info(f"Database engine created: {config.get_database_type()}")
            
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
    
    def initialize_schema(self) -> None:
        """Initialize database schema.
        
        This method ensures the database is ready for data loading.
        Tables will be created dynamically when DataFrames are loaded.
        """
        try:
            # Test connection
            self._test_connection()
            
            # Get existing schema info
            existing_info = self.get_existing_data_info()
            
            self.logger.info(
                f"Schema initialized: {existing_info['total_tables']} existing tables "
                f"with {existing_info['total_rows']} total rows"
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing schema: {e}")
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
                
                # Optimize numeric columns
                if pd.api.types.is_numeric_dtype(col_data):
                    # Try to downcast integers
                    if pd.api.types.is_integer_dtype(col_data):
                        optimized_df[column] = pd.to_numeric(col_data, downcast='integer')
                    # Try to downcast floats
                    elif pd.api.types.is_float_dtype(col_data):
                        optimized_df[column] = pd.to_numeric(col_data, downcast='float')
                
                # Optimize string columns
                elif pd.api.types.is_object_dtype(col_data):
                    # Convert to category if many repeated values
                    unique_ratio = col_data.nunique() / len(col_data)
                    if unique_ratio < 0.5 and col_data.nunique() < 1000:
                        optimized_df[column] = col_data.astype('category')
            
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
        
        # Get schema info
        schema_info = self.get_dataframe_schema_info(optimized_df, table_name)
        
        self.logger.info(
            f"Loading {schema_info['row_count']} rows into table '{table_name}' "
            f"({format_bytes(schema_info['memory_usage_mb'] * 1024 * 1024)})"
        )
        
        def _load_operation():
            # Use chunked loading for large DataFrames
            chunk_size = min(config.CHUNK_SIZE, len(optimized_df))
            
            with self.engine.begin() as conn:  # Transaction context
                optimized_df.to_sql(
                    name=table_name,
                    con=conn,
                    if_exists=if_exists,
                    index=False,
                    chunksize=chunk_size,
                    method='multi'  # Use multi-row INSERT for better performance
                )
        
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
    
    def load_dataframes_batch(self, dataframes_batch: Dict[str, List[pd.DataFrame]], 
                             if_exists: str = 'append') -> None:
        """Load multiple batches of DataFrames (e.g., from multiple files).
        
        Args:
            dataframes_batch: Dictionary mapping table names to lists of DataFrames
            if_exists: How to behave if tables exist ('fail', 'replace', 'append')
        """
        if not dataframes_batch:
            self.logger.warning("No DataFrame batches to load")
            return
        
        # Calculate totals
        total_tables = len(dataframes_batch)
        total_dataframes = sum(len(df_list) for df_list in dataframes_batch.values())
        total_rows = sum(sum(len(df) for df in df_list) for df_list in dataframes_batch.values())
        
        self.logger.info(
            f"Loading {total_tables} table types with {total_dataframes} DataFrames "
            f"and {total_rows} total rows"
        )
        
        successful_tables = []
        failed_tables = []
        
        for table_name, df_list in dataframes_batch.items():
            try:
                # Concatenate all DataFrames for this table
                if len(df_list) == 1:
                    combined_df = df_list[0]
                else:
                    self.logger.info(f"Concatenating {len(df_list)} DataFrames for table {table_name}")
                    combined_df = pd.concat(df_list, ignore_index=True)
                
                # Load the combined DataFrame
                self.load_dataframe(combined_df, table_name, if_exists)
                successful_tables.append(table_name)
                
            except Exception as e:
                self.logger.error(f"Failed to load table {table_name}: {e}")
                failed_tables.append(table_name)
                continue
        
        self.logger.info(
            f"Batch loading complete: {len(successful_tables)} successful, {len(failed_tables)} failed"
        )
        
        if failed_tables:
            self.logger.warning(f"Failed tables: {failed_tables}")
    
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