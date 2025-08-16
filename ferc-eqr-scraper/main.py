"""Main CLI entry point for FERC EQR Scraper."""

import sys
from datetime import datetime
from typing import Optional, Tuple
import click
import logging

from settings import config
from utils import setup_logging, validate_year_range, validate_quarters
from downloader import FERCDownloader
from processor import FERCProcessor
from database import FERCDatabase


def setup_application_logging(verbose: bool = False) -> logging.Logger:
    """Set up application-wide logging.
    
    Args:
        verbose: Enable verbose logging
        
    Returns:
        Main application logger
    """
    log_level = "DEBUG" if verbose else config.LOG_LEVEL
    logger = setup_logging(log_level)
    
    logger.info("=" * 60)
    logger.info("FERC EQR Scraper & Processor")
    logger.info("=" * 60)
    logger.info(f"Log level: {log_level}")
    logger.info(f"Database: {config.get_database_type()}")
    logger.info(f"Download directory: {config.DOWNLOAD_DIR}")
    
    return logger


def validate_date_parameters(start_year: Optional[int], end_year: Optional[int], 
                           quarters: Tuple[str, ...]) -> Tuple[int, int, list[str]]:
    """Validate and normalize date parameters.
    
    Args:
        start_year: Starting year
        end_year: Ending year
        quarters: Tuple of quarters
        
    Returns:
        Tuple of (validated_start_year, validated_end_year, validated_quarters)
        
    Raises:
        click.BadParameter: If parameters are invalid
    """
    try:
        # Validate year range
        validated_start, validated_end = validate_year_range(start_year, end_year)
        
        # Validate quarters
        validated_quarters = validate_quarters(quarters)
        
        return validated_start, validated_end, validated_quarters
        
    except ValueError as e:
        raise click.BadParameter(str(e))


def display_processing_summary(start_year: int, end_year: int, quarters: list[str], 
                             downloaded_files: list[str], processed_data: dict) -> None:
    """Display summary of processing results.
    
    Args:
        start_year: Starting year
        end_year: Ending year
        quarters: List of quarters
        downloaded_files: List of downloaded file paths
        processed_data: Dictionary of processed data
    """
    logger = logging.getLogger("ferc_scraper")
    
    logger.info("=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)
    
    # Date range summary
    if start_year == end_year:
        logger.info(f"Year: {start_year}")
    else:
        logger.info(f"Years: {start_year} - {end_year}")
    logger.info(f"Quarters: {', '.join(quarters)}")
    
    # Download summary
    expected_files = len(range(start_year, end_year + 1)) * len(quarters)
    logger.info(f"Downloads: {len(downloaded_files)}/{expected_files} successful")
    
    # Processing summary
    if processed_data:
        total_tables = len(processed_data)
        total_dataframes = sum(len(df_list) for df_list in processed_data.values())
        total_rows = sum(sum(len(df) for df in df_list) for df_list in processed_data.values())
        
        logger.info(f"Processing: {total_tables} table types, {total_dataframes} DataFrames, {total_rows} rows")
        
        # Table details
        for table_name, df_list in processed_data.items():
            table_rows = sum(len(df) for df in df_list)
            logger.info(f"  {table_name}: {len(df_list)} files, {table_rows} rows")
    else:
        logger.info("Processing: No data processed")
    
    logger.info("=" * 60)


@click.command()
@click.option('--start-year', type=int, 
              help='First year to download data for (e.g., 2020)')
@click.option('--end-year', type=int,
              help='Last year to download data for (e.g., 2023)')
@click.option('--quarters', multiple=True, type=click.Choice(['1', '2', '3', '4']),
              help='Specific quarters to download (can be used multiple times)')
@click.option('--resume', is_flag=True, default=False,
              help='Resume interrupted downloads and processing')
@click.option('--download-only', is_flag=True, default=False,
              help='Only download files, skip processing and database loading')
@click.option('--process-only', is_flag=True, default=False,
              help='Only process existing files, skip downloading')
@click.option('--files', type=int, default=None,
              help='Limit number of XML files to process (useful for testing)')
@click.option('--verbose', '-v', is_flag=True, default=False,
              help='Enable verbose logging')
@click.option('--dry-run', is_flag=True, default=False,
              help='Show what would be done without actually doing it')
def run_pipeline(start_year: Optional[int], end_year: Optional[int], 
                quarters: Tuple[str, ...], resume: bool, download_only: bool,
                process_only: bool, files: Optional[int], verbose: bool, dry_run: bool) -> None:
    """FERC EQR Scraper & Processor - Download and process FERC Electric Quarterly Reports.
    
    This tool downloads FERC EQR data in XML format, processes it using the 
    FERC XBRL extractor, and loads the results into a SQL database.
    
    Examples:
    
        # Download and process all quarters for 2023
        python main.py --start-year 2023 --end-year 2023
        
        # Download specific quarters for multiple years
        python main.py --start-year 2022 --end-year 2023 --quarters 1 --quarters 4
        
        # Resume interrupted processing
        python main.py --start-year 2023 --end-year 2023 --resume
        
        # Only download files (no processing)
        python main.py --start-year 2023 --end-year 2023 --download-only
    """
    # Set up logging
    logger = setup_application_logging(verbose)
    
    try:
        # Validate parameters
        if not start_year and not end_year:
            raise click.BadParameter("Either --start-year or --end-year must be specified")
        
        if download_only and process_only:
            raise click.BadParameter("Cannot specify both --download-only and --process-only")
        
        # Validate and normalize date parameters
        validated_start, validated_end, validated_quarters = validate_date_parameters(
            start_year, end_year, quarters
        )
        
        if dry_run:
            logger.info("DRY RUN MODE - No actual operations will be performed")
        
        # Display configuration
        logger.info(f"Processing years {validated_start}-{validated_end}, quarters: {validated_quarters}")
        logger.info(f"Resume mode: {'enabled' if resume else 'disabled'}")
        
        if download_only:
            logger.info("Mode: Download only")
        elif process_only:
            logger.info("Mode: Process only")
        else:
            logger.info("Mode: Full pipeline (download + process + load)")
        
        # Initialize components
        downloader = FERCDownloader() if not process_only else None
        processor = FERCProcessor() if not download_only else None
        database = FERCDatabase() if not download_only else None
        
        downloaded_files = []
        processed_data = {}
        
        # Step 1: Download files
        if not process_only:
            logger.info("Starting download phase...")
            
            if dry_run:
                urls = downloader.generate_urls(validated_start, validated_end, validated_quarters)
                logger.info(f"Would download {len(urls)} files:")
                for url in urls:
                    logger.info(f"  {url}")
            else:
                urls = downloader.generate_urls(validated_start, validated_end, validated_quarters)
                downloaded_files = downloader.download_multiple(urls, resume=resume)
                
                if not downloaded_files:
                    logger.error("No files were downloaded successfully")
                    sys.exit(1)
        
        # Step 2: Process files
        if not download_only:
            logger.info("Starting processing phase...")
            
            if process_only:
                # Find existing files to process
                import glob
                import os
                pattern = os.path.join(config.DOWNLOAD_DIR, "XML_*.zip")
                downloaded_files = glob.glob(pattern)
                
                if not downloaded_files:
                    logger.error(f"No ZIP files found in {config.DOWNLOAD_DIR}")
                    sys.exit(1)
                
                logger.info(f"Found {len(downloaded_files)} existing files to process")
            
            if dry_run:
                logger.info(f"Would process {len(downloaded_files)} files")
            else:
                processed_data = processor.process_multiple_files(downloaded_files, files)
                
                if not processed_data:
                    logger.warning("No data was extracted from the files")
        
        # Step 3: Load to database
        if not download_only and not dry_run:
            if processed_data:
                logger.info("Starting database loading phase...")
                
                # Initialize database schema
                database.initialize_schema()
                
                # Load the data
                database.load_dataframes_batch(processed_data)
                
                # Get final database info
                final_info = database.get_existing_data_info()
                logger.info(f"Database now contains {final_info['total_rows']} total rows across {final_info['total_tables']} tables")
            else:
                logger.info("No data to load to database")
        
        # Display summary
        if not dry_run:
            display_processing_summary(
                validated_start, validated_end, validated_quarters,
                downloaded_files, processed_data
            )
        
        logger.info("Pipeline completed successfully!")
        
    except click.BadParameter as e:
        logger.error(f"Parameter error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # Clean up resources
        if 'database' in locals() and database:
            database.close()


if __name__ == "__main__":
    run_pipeline()