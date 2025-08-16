"""Download component for FERC EQR data files."""

import os
import time
from pathlib import Path
from typing import List, Optional
import requests
from tqdm import tqdm
import logging

from settings import config
from utils import ensure_directory, get_file_size, format_bytes


class FERCDownloader:
    """High-performance downloader for FERC EQR files."""
    
    def __init__(self):
        """Initialize the downloader."""
        self.logger = logging.getLogger("ferc_scraper.downloader")
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FERC-EQR-Scraper/1.0 (Python/requests)'
        })
        
        # Ensure download directory exists
        ensure_directory(config.DOWNLOAD_DIR)
    
    def generate_urls(self, start_year: int, end_year: int, quarters: List[str]) -> List[str]:
        """Generate FERC EQR download URLs for specified years and quarters.
        
        Args:
            start_year: Starting year for downloads
            end_year: Ending year for downloads
            quarters: List of quarters to download (1-4)
            
        Returns:
            List of download URLs
        """
        urls = []
        
        for year in range(start_year, end_year + 1):
            for quarter in quarters:
                filename = f"XML_{year}_Q{quarter}.zip"
                url = f"{config.FERC_BASE_URL}{filename}"
                urls.append(url)
        
        self.logger.info(f"Generated {len(urls)} download URLs for years {start_year}-{end_year}, quarters {quarters}")
        return urls
    
    def get_local_filename(self, url: str) -> str:
        """Get the local filename for a download URL.
        
        Args:
            url: Download URL
            
        Returns:
            Local file path
        """
        filename = url.split('/')[-1]
        return os.path.join(config.DOWNLOAD_DIR, filename)
    
    def verify_download(self, filepath: str, expected_size: Optional[int] = None) -> bool:
        """Verify that a downloaded file is complete and valid.
        
        Args:
            filepath: Path to the downloaded file
            expected_size: Expected file size in bytes (optional)
            
        Returns:
            True if file is valid, False otherwise
        """
        if not os.path.exists(filepath):
            return False
        
        file_size = get_file_size(filepath)
        if file_size == 0:
            return False
        
        if expected_size is not None and file_size != expected_size:
            self.logger.warning(f"File size mismatch: expected {expected_size}, got {file_size}")
            return False
        
        # Basic ZIP file validation - check for ZIP signature
        try:
            with open(filepath, 'rb') as f:
                signature = f.read(4)
                if signature != b'PK\x03\x04':  # ZIP file signature
                    self.logger.warning(f"Invalid ZIP signature in {filepath}")
                    return False
        except Exception as e:
            self.logger.error(f"Error validating file {filepath}: {e}")
            return False
        
        return True
    
    def download_file(self, url: str, resume: bool = True) -> str:
        """Download a file with progress tracking and resume capability.
        
        Args:
            url: URL to download
            resume: Whether to resume interrupted downloads
            
        Returns:
            Path to the downloaded file
            
        Raises:
            requests.RequestException: If download fails after all retries
        """
        local_filename = self.get_local_filename(url)
        
        # Check if file already exists and is valid
        if os.path.exists(local_filename) and self.verify_download(local_filename):
            self.logger.info(f"File already exists and is valid: {local_filename}")
            return local_filename
        
        # Determine resume position
        resume_pos = 0
        if resume and os.path.exists(local_filename):
            resume_pos = get_file_size(local_filename)
            self.logger.info(f"Resuming download from position {format_bytes(resume_pos)}")
        
        # Prepare headers for resume
        headers = {}
        if resume_pos > 0:
            headers['Range'] = f'bytes={resume_pos}-'
        
        # Attempt download with retries
        for attempt in range(config.MAX_RETRIES + 1):
            try:
                return self._download_with_progress(url, local_filename, headers, resume_pos)
            except Exception as e:
                if attempt < config.MAX_RETRIES:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(f"Download attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Download failed after {config.MAX_RETRIES + 1} attempts: {e}")
                    raise
    
    def _download_with_progress(self, url: str, local_filename: str, headers: dict, resume_pos: int) -> str:
        """Internal method to download file with progress bar.
        
        Args:
            url: URL to download
            local_filename: Local file path
            headers: HTTP headers (for resume)
            resume_pos: Position to resume from
            
        Returns:
            Path to downloaded file
        """
        response = self.session.get(
            url, 
            headers=headers, 
            stream=True, 
            timeout=config.REQUEST_TIMEOUT
        )
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        if resume_pos > 0:
            total_size += resume_pos
        
        # Open file in appropriate mode
        mode = 'ab' if resume_pos > 0 else 'wb'
        
        with open(local_filename, mode) as f:
            with tqdm(
                total=total_size,
                initial=resume_pos,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {os.path.basename(local_filename)}"
            ) as pbar:
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Verify the download
        if not self.verify_download(local_filename, total_size):
            raise ValueError(f"Downloaded file failed verification: {local_filename}")
        
        self.logger.info(f"Successfully downloaded: {local_filename} ({format_bytes(total_size)})")
        return local_filename
    
    def download_multiple(self, urls: List[str], resume: bool = True) -> List[str]:
        """Download multiple files sequentially.
        
        Args:
            urls: List of URLs to download
            resume: Whether to resume interrupted downloads
            
        Returns:
            List of downloaded file paths
        """
        downloaded_files = []
        
        self.logger.info(f"Starting download of {len(urls)} files")
        
        for i, url in enumerate(urls, 1):
            self.logger.info(f"Downloading file {i}/{len(urls)}: {url}")
            try:
                filepath = self.download_file(url, resume)
                downloaded_files.append(filepath)
            except Exception as e:
                self.logger.error(f"Failed to download {url}: {e}")
                # Continue with other downloads
                continue
        
        self.logger.info(f"Download complete: {len(downloaded_files)}/{len(urls)} files successful")
        return downloaded_files
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'session'):
            self.session.close()