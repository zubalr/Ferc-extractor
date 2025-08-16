"""Configuration management for FERC EQR Scraper."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging


class Config:
    """Configuration class for FERC EQR Scraper."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        # Directory settings
        self.DOWNLOAD_DIR = "ferc_data/downloads"
        self.EXTRACT_DIR = "ferc_data/extracted"
        
        # Database settings
        self.DATABASE_URI = os.environ.get("FERC_DATABASE_URI", "sqlite:///ferc_data.db")
        
        # Logging settings
        self.LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
        
        # Performance settings
        self.CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "10000"))
        self.MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
        
        # Network settings
        self.REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "30"))
        
        # FERC URL settings
        self.FERC_BASE_URL = "https://eqrreportviewer.ferc.gov/DownloadRepositoryProd/5156CA313D8F46BE81819F62F761AC20A8895E197E1D4299ABFF472FE47F880F88FA8317470A4D0FBFB18CB3274BB531/BulkNew/XML/"
        
        # Validate and initialize
        self._validate_settings()
        self._ensure_directories()
    
    def _validate_settings(self):
        """Validate configuration settings."""
        if self.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if self.MAX_RETRIES < 0:
            raise ValueError("MAX_RETRIES must be non-negative")
        if self.REQUEST_TIMEOUT <= 0:
            raise ValueError("REQUEST_TIMEOUT must be positive")
        
        # Validate log level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.LOG_LEVEL.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of: {', '.join(valid_levels)}")
        
        # Validate database URI format
        if not self.DATABASE_URI:
            raise ValueError("DATABASE_URI cannot be empty")
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        Path(self.DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.EXTRACT_DIR).mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
    
    def get_database_type(self) -> str:
        """Get the database type from the URI."""
        if self.DATABASE_URI.startswith("sqlite"):
            return "sqlite"
        elif self.DATABASE_URI.startswith("postgresql"):
            return "postgresql"
        else:
            return "unknown"
    
    def is_postgresql(self) -> bool:
        """Check if using PostgreSQL database."""
        return self.get_database_type() == "postgresql"
    
    def is_sqlite(self) -> bool:
        """Check if using SQLite database."""
        return self.get_database_type() == "sqlite"


def load_config() -> Config:
    """Load and validate configuration."""
    try:
        return Config()
    except Exception as e:
        print(f"Configuration error: {e}")
        print("\nExample configuration:")
        print("export FERC_DATABASE_URI='sqlite:///ferc_data.db'")
        print("export LOG_LEVEL='INFO'")
        print("export CHUNK_SIZE='10000'")
        raise


# Global configuration instance
config = load_config()