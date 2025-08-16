"""Utility functions and helpers for FERC EQR Scraper."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                logs_dir / f"ferc_scraper_{datetime.now().strftime('%Y%m%d')}.log"
            )
        ]
    )
    
    return logging.getLogger("ferc_scraper")


def ensure_directory(directory: str) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to create
        
    Returns:
        Path object for the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(filepath: str) -> int:
    """Get file size in bytes.
    
    Args:
        filepath: Path to the file
        
    Returns:
        File size in bytes, or 0 if file doesn't exist
    """
    try:
        return os.path.getsize(filepath)
    except (OSError, FileNotFoundError):
        return 0


def format_bytes(bytes_count: int) -> str:
    """Format bytes into human-readable string.
    
    Args:
        bytes_count: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f} PB"


def validate_year_range(start_year: Optional[int], end_year: Optional[int]) -> tuple[int, int]:
    """Validate and normalize year range.
    
    Args:
        start_year: Starting year
        end_year: Ending year
        
    Returns:
        Tuple of (start_year, end_year)
        
    Raises:
        ValueError: If year range is invalid
    """
    current_year = datetime.now().year
    
    if start_year is None and end_year is None:
        raise ValueError("Either start_year or end_year must be specified")
    
    if start_year is None:
        start_year = end_year
    if end_year is None:
        end_year = start_year
        
    if start_year > end_year:
        raise ValueError("start_year cannot be greater than end_year")
    
    if start_year < 2000:
        raise ValueError("start_year must be 2000 or later")
    
    if end_year > current_year:
        raise ValueError(f"end_year cannot be greater than current year ({current_year})")
    
    return start_year, end_year


def validate_quarters(quarters: tuple[str, ...]) -> list[str]:
    """Validate and normalize quarters.
    
    Args:
        quarters: Tuple of quarter strings
        
    Returns:
        List of valid quarter strings
        
    Raises:
        ValueError: If quarters are invalid
    """
    if not quarters:
        return ['1', '2', '3', '4']  # Default to all quarters
    
    valid_quarters = {'1', '2', '3', '4'}
    quarters_list = list(quarters)
    
    for quarter in quarters_list:
        if quarter not in valid_quarters:
            raise ValueError(f"Invalid quarter: {quarter}. Must be 1, 2, 3, or 4")
    
    return sorted(list(set(quarters_list)))  # Remove duplicates and sort