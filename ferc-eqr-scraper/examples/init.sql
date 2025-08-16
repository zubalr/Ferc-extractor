-- PostgreSQL initialization script for FERC EQR Scraper
-- This script sets up the database with appropriate permissions

-- Create the database (if not already created by Docker)
-- CREATE DATABASE ferc_data;

-- Connect to the database
\c ferc_data;

-- Create extensions that might be useful for data analysis
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create a schema for FERC data (optional - tables will be created in public by default)
-- CREATE SCHEMA IF NOT EXISTS ferc;

-- Grant permissions to the ferc_user
GRANT ALL PRIVILEGES ON DATABASE ferc_data TO ferc_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO ferc_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ferc_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ferc_user;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO ferc_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO ferc_user;

-- Create a function to get table sizes (useful for monitoring)
CREATE OR REPLACE FUNCTION get_table_sizes()
RETURNS TABLE(
    table_name text,
    row_count bigint,
    total_size text,
    table_size text,
    index_size text
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        schemaname||'.'||tablename as table_name,
        n_tup_ins - n_tup_del as row_count,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
        pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) as index_size
    FROM pg_stat_user_tables
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
END;
$$ LANGUAGE plpgsql;

-- Create a view for monitoring data loading progress
CREATE OR REPLACE VIEW data_loading_stats AS
SELECT 
    schemaname,
    tablename,
    n_tup_ins as rows_inserted,
    n_tup_upd as rows_updated,
    n_tup_del as rows_deleted,
    n_live_tup as current_rows,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;

-- Print completion message
\echo 'FERC EQR database initialization completed successfully!'
\echo 'Use SELECT * FROM get_table_sizes(); to monitor table sizes'
\echo 'Use SELECT * FROM data_loading_stats; to monitor data loading progress'