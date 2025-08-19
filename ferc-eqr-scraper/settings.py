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
        
        # Database settings - Production ready for Turso
        self.DATABASE_URI = os.environ.get("FERC_DATABASE_URI", "sqlite:///ferc_data.db")
        self.TURSO_DATABASE_URL = os.environ.get("TURSO_DATABASE_URL")
        self.TURSO_AUTH_TOKEN = os.environ.get("TURSO_AUTH_TOKEN")
        
        # Use Turso if available, otherwise fall back to local SQLite
        if self.TURSO_DATABASE_URL and self.TURSO_AUTH_TOKEN:
            self.DATABASE_URI = f"sqlite+libsql://{self.TURSO_DATABASE_URL}?authToken={self.TURSO_AUTH_TOKEN}"
        
        # Logging settings
        self.LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
        
        # Performance settings
        self.CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "5000"))  # Reduced for better memory management
        self.MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
        
        # Memory management settings
        self.MAX_MEMORY_USAGE_PCT = float(os.environ.get("MAX_MEMORY_USAGE_PCT", "95.0"))  # Increased to 95% for better throughput
        self.BATCH_SIZE_XML_FILES = int(os.environ.get("BATCH_SIZE_XML_FILES", "5"))  # Reduced from 10 to 5
        self.CONCAT_CHUNK_SIZE = int(os.environ.get("CONCAT_CHUNK_SIZE", "10"))  # Reduced from 25 to 10
        
        # Large file handling (in bytes)
        self.MAX_TRANSACTION_ROWS_MEMORY = int(os.environ.get("MAX_TRANSACTION_ROWS_MEMORY", "50000"))  # Max transactions to hold in memory
        self.LARGE_FILE_STREAMING_THRESHOLD = int(os.environ.get("LARGE_FILE_STREAMING_THRESHOLD", "20")) * 1024 * 1024  # 20MB instead of 50MB
        self.EMERGENCY_MEMORY_THRESHOLD = float(os.environ.get("EMERGENCY_MEMORY_THRESHOLD", "80.0"))  # Stop processing if memory exceeds 80%
        
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
        if "libsql://" in self.DATABASE_URI:
            return "turso"
        elif self.DATABASE_URI.startswith("sqlite"):
            return "sqlite"
        elif self.DATABASE_URI.startswith("postgresql"):
            return "postgresql"
        else:
            return "unknown"
    
    def is_turso(self) -> bool:
        """Check if using Turso database."""
        return self.get_database_type() == "turso"
    
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