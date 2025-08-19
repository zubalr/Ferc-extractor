"""Processing component for FERC EQR XML data extraction with memory efficiency."""

import os
import shutil
import zipfile
import gc
import psutil
from pathlib import Path
from typing import Dict, Optional, Generator, Iterator
import pandas as pd
import logging
from tqdm import tqdm

from settings import config
from utils import ensure_directory, format_bytes, get_file_size
from eqr_parser import EQRXMLParser


class FERCProcessor:
    """Memory-efficient processor for FERC EQR ZIP files and XML data extraction."""
    
    def __init__(self, max_memory_usage_pct: float = 70.0):
        """Initialize the processor with memory monitoring.
        
        Args:
            max_memory_usage_pct: Maximum memory usage percentage before triggering cleanup
        """
        self.logger = logging.getLogger("ferc_scraper.processor")
        self.eqr_parser = EQRXMLParser()
        self.max_memory_usage_pct = max_memory_usage_pct
        
        # Memory monitoring
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
        
        # Ensure directories exist
        ensure_directory(config.EXTRACT_DIR)
        
        self.logger.info(f"Initialized processor with {max_memory_usage_pct}% memory limit")
        self.logger.info(f"Initial memory usage: {format_bytes(self.initial_memory)}")
    
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return self.process.memory_info().rss

    def get_memory_usage_pct(self) -> float:
        """Get current memory usage as percentage of total system memory."""
        return psutil.virtual_memory().percent

    def monitor_memory(self, operation_name: str) -> None:
        """Monitor and log memory usage for an operation."""
        current_memory = self.get_memory_usage()
        current_pct = self.get_memory_usage_pct()
        
        self.logger.debug(
            f"Memory usage during {operation_name}: {format_bytes(current_memory)} ({current_pct:.1f}%)"
        )
        
        # Force garbage collection if memory usage is high
        if current_pct > self.max_memory_usage_pct:
            self.logger.warning(f"High memory usage ({current_pct:.1f}%), forcing garbage collection")
            collected = gc.collect()
            new_memory = self.get_memory_usage()
            new_pct = self.get_memory_usage_pct()
            
            freed_memory = current_memory - new_memory
            self.logger.info(
                f"Garbage collection: freed {collected} objects, "
                f"recovered {format_bytes(freed_memory)} "
                f"(now {new_pct:.1f}%)"
            )

    def check_emergency_memory_limit(self) -> bool:
        """Check if memory usage has exceeded emergency threshold.
        
        Returns:
            True if emergency threshold exceeded, False otherwise
        """
        current_pct = self.get_memory_usage_pct()
        emergency_threshold = config.EMERGENCY_MEMORY_THRESHOLD
        
        if current_pct > emergency_threshold:
            self.logger.error(f"EMERGENCY: Memory usage ({current_pct:.1f}%) exceeded threshold ({emergency_threshold}%)")
            return True
        return False

    def should_process_more(self) -> bool:
        """Check if we should continue processing based on memory usage."""
        current_pct = self.get_memory_usage_pct()
        
        # Emergency stop - if memory is too high, stop immediately
        if self.check_emergency_memory_limit():
            self.logger.error("EMERGENCY STOP: Memory usage exceeded emergency threshold")
            return False
            
        # Normal memory limit check
        if current_pct > self.max_memory_usage_pct:
            self.logger.warning(f"Memory usage too high ({current_pct:.1f}%), should pause processing")
            return False
        return True
    
    def extract_archive(self, zip_path: str, extract_dir: Optional[str] = None) -> str:
        """Extract ZIP archive to specified directory.
        
        Args:
            zip_path: Path to ZIP file
            extract_dir: Directory to extract to (defaults to config.EXTRACT_DIR)
            
        Returns:
            Path to extraction directory
            
        Raises:
            zipfile.BadZipFile: If ZIP file is corrupted
            OSError: If extraction fails
        """
        if extract_dir is None:
            extract_dir = config.EXTRACT_DIR
        
        # Clean up any existing extraction
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        
        ensure_directory(extract_dir)
        
        self.logger.info(f"Extracting {zip_path} to {extract_dir}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files to extract
                file_list = zip_ref.namelist()
                
                # Extract with progress bar
                with tqdm(total=len(file_list), desc="Extracting files") as pbar:
                    for file_info in zip_ref.infolist():
                        zip_ref.extract(file_info, extract_dir)
                        pbar.update(1)
                
                self.logger.info(f"Extracted {len(file_list)} files to {extract_dir}")
                
        except zipfile.BadZipFile as e:
            self.logger.error(f"Invalid ZIP file {zip_path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error extracting {zip_path}: {e}")
            raise
        
        return extract_dir
    
    def get_extracted_files(self, extract_dir: str) -> list[str]:
        """Get list of extracted files.
        
        Args:
            extract_dir: Directory containing extracted files
            
        Returns:
            List of file paths
        """
        files = []
        for root, dirs, filenames in os.walk(extract_dir):
            for filename in filenames:
                files.append(os.path.join(root, filename))
        
        self.logger.info(f"Found {len(files)} extracted files")
        return files
    
    def validate_extracted_files(self, extract_dir: str) -> bool:
        """Validate that extracted files are present and readable.
        
        Args:
            extract_dir: Directory containing extracted files
            
        Returns:
            True if files are valid, False otherwise
        """
        try:
            files = self.get_extracted_files(extract_dir)
            
            if not files:
                self.logger.warning(f"No files found in {extract_dir}")
                return False
            
            # Check for XML files (direct XML files)
            xml_files = [f for f in files if f.lower().endswith('.xml')]
            
            # Check for ZIP files (nested ZIP files containing XML)
            zip_files = [f for f in files if f.lower().endswith('.zip')]
            
            if not xml_files and not zip_files:
                self.logger.warning(f"No XML or ZIP files found in {extract_dir}")
                return False
            
            # If we have direct XML files, validate them
            if xml_files:
                for file_path in xml_files[:5]:  # Check first 5 XML files
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            f.read(100)  # Read first 100 characters
                    except Exception as e:
                        self.logger.warning(f"Cannot read XML file {file_path}: {e}")
                        return False
                
                self.logger.info(f"Validation successful: {len(xml_files)} XML files found")
                return True
            
            # If we have ZIP files, check if they contain XML files
            if zip_files:
                valid_zip_count = 0
                for zip_path in zip_files[:10]:  # Check first 10 ZIP files
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_contents = zip_ref.namelist()
                            xml_in_zip = [f for f in zip_contents if f.lower().endswith('.xml')]
                            if xml_in_zip:
                                valid_zip_count += 1
                    except Exception as e:
                        self.logger.debug(f"Cannot read ZIP file {zip_path}: {e}")
                        continue
                
                if valid_zip_count > 0:
                    self.logger.info(f"Validation successful: {len(zip_files)} ZIP files found, {valid_zip_count} contain XML files")
                    return True
                else:
                    self.logger.warning(f"No valid XML files found in ZIP files")
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error validating extracted files: {e}")
            return False
    
    def cleanup_temp_files(self, extract_dir: str) -> None:
        """Clean up temporary extracted files.
        
        Args:
            extract_dir: Directory to clean up
        """
        try:
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
                self.logger.info(f"Cleaned up temporary directory: {extract_dir}")
        except Exception as e:
            self.logger.warning(f"Error cleaning up {extract_dir}: {e}")
    
    def get_extraction_stats(self, extract_dir: str) -> Dict[str, int]:
        """Get statistics about extracted files.
        
        Args:
            extract_dir: Directory containing extracted files
            
        Returns:
            Dictionary with file statistics
        """
        stats = {
            'total_files': 0,
            'xml_files': 0,
            'zip_files': 0,
            'total_size': 0,
            'xml_size': 0,
            'zip_size': 0
        }
        
        try:
            for root, dirs, filenames in os.walk(extract_dir):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    file_size = get_file_size(file_path)
                    
                    stats['total_files'] += 1
                    stats['total_size'] += file_size
                    
                    if filename.lower().endswith('.xml'):
                        stats['xml_files'] += 1
                        stats['xml_size'] += file_size
                    elif filename.lower().endswith('.zip'):
                        stats['zip_files'] += 1
                        stats['zip_size'] += file_size
            
            self.logger.info(
                f"Extraction stats: {stats['total_files']} files "
                f"({format_bytes(stats['total_size'])}), "
                f"{stats['xml_files']} XML files "
                f"({format_bytes(stats['xml_size'])}), "
                f"{stats['zip_files']} ZIP files "
                f"({format_bytes(stats['zip_size'])})"
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating extraction stats: {e}")
        
        return stats
    
    def extract_nested_zip_files(self, extract_dir: str, max_files: Optional[int] = None) -> str:
        """Extract nested ZIP files to get XML files.
        
        Args:
            extract_dir: Directory containing nested ZIP files
            max_files: Maximum number of nested ZIP files to extract (None for all)
            
        Returns:
            Directory containing extracted XML files
        """
        xml_extract_dir = os.path.join(extract_dir, "xml_files")
        ensure_directory(xml_extract_dir)
        
        zip_files = [f for f in self.get_extracted_files(extract_dir) if f.lower().endswith('.zip')]
        
        # Limit files if requested
        if max_files is not None and max_files < len(zip_files):
            zip_files = zip_files[:max_files]
            self.logger.info(f"Limited to first {max_files} nested ZIP files for processing")
        
        self.logger.info(f"Extracting {len(zip_files)} nested ZIP files...")
        
        extracted_count = 0
        with tqdm(total=len(zip_files), desc="Extracting nested ZIP files") as pbar:
            for zip_path in zip_files:
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        # Extract to a subdirectory named after the ZIP file
                        zip_name = os.path.splitext(os.path.basename(zip_path))[0]
                        zip_extract_path = os.path.join(xml_extract_dir, zip_name)
                        ensure_directory(zip_extract_path)
                        
                        zip_ref.extractall(zip_extract_path)
                        extracted_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Failed to extract nested ZIP {zip_path}: {e}")
                    continue
                finally:
                    pbar.update(1)
        
        self.logger.info(f"Successfully extracted {extracted_count}/{len(zip_files)} nested ZIP files")
        return xml_extract_dir

    def parse_xml_data(self, extract_dir: str, max_files: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Parse XML data using custom EQR parser with memory efficiency.
        
        Args:
            extract_dir: Directory containing extracted files
            max_files: Maximum number of nested ZIP files to process (None for all)
            
        Returns:
            Dictionary of DataFrames with parsed data
            
        Raises:
            Exception: If parsing fails
        """
        self.logger.info("Starting FERC EQR XML data extraction...")
        self.monitor_memory("XML parsing start")
        
        xml_extract_dir = None
        try:
            # Check if we have nested ZIP files that need to be extracted
            zip_files = [f for f in self.get_extracted_files(extract_dir) if f.lower().endswith('.zip')]
            xml_files = [f for f in self.get_extracted_files(extract_dir) if f.lower().endswith('.xml')]
            
            if zip_files and not xml_files:
                # We have nested ZIP files, extract them first
                xml_extract_dir = self.extract_nested_zip_files(extract_dir, max_files)
                # Find all XML files in the extracted directory
                xml_files = []
                for root, dirs, filenames in os.walk(xml_extract_dir):
                    for filename in filenames:
                        if filename.lower().endswith('.xml'):
                            xml_files.append(os.path.join(root, filename))
            else:
                xml_extract_dir = extract_dir
            
            if not xml_files:
                self.logger.warning("No XML files found for processing")
                return {}
            
            self.logger.info(f"Found {len(xml_files)} XML files to process")
            self.logger.info(f"XML files are located in: {xml_extract_dir}")
            
            # Use our custom EQR parser with streaming approach
            self.logger.info("Parsing EQR XML data using custom parser...")
            
            # Parse XML files in batches to manage memory
            batch_size = min(10, len(xml_files))  # Process 10 files at a time
            combined_dataframes = {}
            processed_files = 0
            
            for batch_start in range(0, len(xml_files), batch_size):
                batch_end = min(batch_start + batch_size, len(xml_files))
                batch_files = xml_files[batch_start:batch_end]
                
                self.logger.info(f"Processing batch {batch_start//batch_size + 1}: files {batch_start+1}-{batch_end}")
                
                # Check memory before processing batch
                if not self.should_process_more():
                    self.logger.warning("Memory usage too high, stopping processing")
                    break
                
                # Parse batch of files
                batch_dataframes = self.eqr_parser.parse_multiple_files(batch_files)
                
                # Merge with existing DataFrames
                for table_name, df_list in batch_dataframes.items():
                    if table_name not in combined_dataframes:
                        combined_dataframes[table_name] = []
                    combined_dataframes[table_name].extend(df_list)
                
                processed_files += len(batch_files)
                self.monitor_memory(f"batch processing {processed_files}/{len(xml_files)} files")
                
                # Force cleanup after each batch
                del batch_dataframes
                gc.collect()
            
            if not combined_dataframes:
                self.logger.warning("No data extracted from XML files")
                return {}
            
            # Convert lists to single DataFrames efficiently
            final_dataframes = {}
            for table_name, df_list in combined_dataframes.items():
                if df_list:
                    if len(df_list) == 1:
                        final_dataframes[table_name] = df_list[0]
                    else:
                        # Process concatenation in chunks if too many DataFrames
                        if len(df_list) > 50:  # Arbitrary threshold
                            self.logger.info(f"Large number of DataFrames for {table_name} ({len(df_list)}), processing in chunks")
                            
                            # Concatenate in chunks to avoid memory issues
                            chunk_size = 25
                            chunks = []
                            for i in range(0, len(df_list), chunk_size):
                                chunk_dfs = df_list[i:i+chunk_size]
                                chunk_combined = pd.concat(chunk_dfs, ignore_index=True)
                                chunks.append(chunk_combined)
                                
                                # Clean up chunk DataFrames
                                del chunk_dfs
                                gc.collect()
                            
                            # Final concatenation
                            final_dataframes[table_name] = pd.concat(chunks, ignore_index=True)
                            del chunks
                        else:
                            final_dataframes[table_name] = pd.concat(df_list, ignore_index=True)
                    
                    self.logger.info(f"Final {table_name}: {len(final_dataframes[table_name])} total rows")
                    self.monitor_memory(f"combined {table_name}")
            
            # Clean up intermediate data
            del combined_dataframes
            gc.collect()
            
            # Log extraction results
            total_rows = sum(len(df) for df in final_dataframes.values())
            self.logger.info(
                f"Successfully extracted {len(final_dataframes)} tables "
                f"with {total_rows} total rows from {processed_files} files"
            )
            
            # Log table details
            for table_name, df in final_dataframes.items():
                memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                self.logger.info(f"  {table_name}: {len(df)} rows, {len(df.columns)} columns, {memory_mb:.1f} MB")
            
            return final_dataframes
                
        except Exception as e:
            self.logger.error(f"Error during EQR XML extraction: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            # Always force cleanup
            gc.collect()
    
    def validate_dataframes(self, dataframes: Dict[str, pd.DataFrame]) -> bool:
        """Validate extracted DataFrames.
        
        Args:
            dataframes: Dictionary of DataFrames to validate
            
        Returns:
            True if DataFrames are valid, False otherwise
        """
        if not dataframes:
            self.logger.warning("No DataFrames to validate")
            return False
        
        try:
            for table_name, df in dataframes.items():
                if df.empty:
                    self.logger.warning(f"Table {table_name} is empty")
                    continue
                
                # Check for basic data integrity
                if df.isnull().all().all():
                    self.logger.warning(f"Table {table_name} contains only null values")
                    continue
                
                # Log basic statistics
                self.logger.debug(f"Table {table_name}: {len(df)} rows, {df.memory_usage(deep=True).sum()} bytes")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating DataFrames: {e}")
            return False
    
    def process_zipfile(self, zip_path: str, max_files: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Process a FERC EQR ZIP file and extract data.
        
        Args:
            zip_path: Path to ZIP file to process
            max_files: Maximum number of nested ZIP files to process (None for all)
            
        Returns:
            Dictionary of DataFrames with extracted data
            
        Raises:
            Exception: If processing fails
        """
        self.logger.info(f"Processing ZIP file: {zip_path}")
        
        # Extract the archive
        extract_dir = self.extract_archive(zip_path)
        
        try:
            # Validate extracted files
            if not self.validate_extracted_files(extract_dir):
                raise ValueError(f"Invalid or missing files in {zip_path}")
            
            # Get extraction statistics
            stats = self.get_extraction_stats(extract_dir)
            
            # Parse XML data using FERC XBRL extractor
            dataframes = self.parse_xml_data(extract_dir, max_files)
            
            # Validate extracted data
            if not self.validate_dataframes(dataframes):
                self.logger.warning(f"Data validation issues in {zip_path}")
            
            return dataframes
            
        except Exception as e:
            self.logger.error(f"Error processing {zip_path}: {e}")
            raise
        finally:
            # Always clean up temporary files
            self.cleanup_temp_files(extract_dir)
    
    def handle_processing_error(self, zip_path: str, error: Exception) -> bool:
        """Handle processing errors with recovery options.
        
        Args:
            zip_path: Path to the ZIP file that failed
            error: The exception that occurred
            
        Returns:
            True if error was handled and processing should continue, False otherwise
        """
        error_type = type(error).__name__
        
        if isinstance(error, zipfile.BadZipFile):
            self.logger.error(f"Corrupted ZIP file {zip_path}: {error}")
            return True  # Skip this file and continue
        
        elif isinstance(error, ImportError):
            self.logger.error(f"Missing dependency for {zip_path}: {error}")
            return False  # Cannot continue without required library
        
        elif isinstance(error, MemoryError):
            self.logger.error(f"Out of memory processing {zip_path}: {error}")
            # Try to free up memory
            import gc
            gc.collect()
            return True  # Skip this file and continue
        
        elif isinstance(error, OSError):
            self.logger.error(f"File system error processing {zip_path}: {error}")
            return True  # Skip this file and continue
        
        else:
            self.logger.error(f"Unexpected error processing {zip_path} ({error_type}): {error}")
            return True  # Skip this file and continue by default
    
    def process_multiple_files(self, zip_paths: list[str], max_files: Optional[int] = None) -> Dict[str, list[pd.DataFrame]]:
        """Process multiple ZIP files with error handling and memory management.
        
        Args:
            zip_paths: List of ZIP file paths to process
            max_files: Maximum number of nested ZIP files to process per main ZIP (None for all)
            
        Returns:
            Dictionary mapping table names to lists of DataFrames
        """
        all_dataframes = {}
        successful_files = 0
        failed_files = 0
        
        self.logger.info(f"Processing {len(zip_paths)} ZIP files")
        self.monitor_memory("processing start")
        
        for i, zip_path in enumerate(zip_paths, 1):
            self.logger.info(f"Processing file {i}/{len(zip_paths)}: {os.path.basename(zip_path)}")
            
            # Check memory before processing each file
            if not self.should_process_more():
                self.logger.warning(f"Memory usage too high, stopping processing at file {i}")
                break
            
            try:
                dataframes = self.process_zipfile(zip_path, max_files)
                
                # Merge dataframes by table name - but keep them as separate DataFrames
                # to avoid memory issues from large concatenations
                for table_name, df in dataframes.items():
                    if table_name not in all_dataframes:
                        all_dataframes[table_name] = []
                    all_dataframes[table_name].append(df)
                
                successful_files += 1
                self.monitor_memory(f"processed file {i}")
                
                # Force cleanup after each file
                del dataframes
                gc.collect()
                    
            except Exception as e:
                failed_files += 1
                
                # Handle the error and decide whether to continue
                if not self.handle_processing_error(zip_path, e):
                    self.logger.error("Critical error encountered, stopping processing")
                    break
                
                # Continue with other files
                continue
        
        self.logger.info(
            f"Processing complete: {successful_files} successful, {failed_files} failed. "
            f"Found {len(all_dataframes)} table types."
        )
        
        # Final memory cleanup
        final_memory = self.get_memory_usage()
        memory_delta = final_memory - self.initial_memory
        self.logger.info(
            f"Final memory usage: {format_bytes(final_memory)} "
            f"[Δ{format_bytes(memory_delta, signed=True)} from start]"
        )
        
        return all_dataframes
    
    def get_processing_summary(self, dataframes: Dict[str, list[pd.DataFrame]]) -> Dict[str, int]:
        """Get summary statistics for processed data.
        
        Args:
            dataframes: Dictionary of processed DataFrames
            
        Returns:
            Dictionary with processing statistics
        """
        summary = {
            'total_tables': len(dataframes),
            'total_dataframes': sum(len(df_list) for df_list in dataframes.values()),
            'total_rows': 0,
            'total_memory_mb': 0
        }
        
        try:
            for table_name, df_list in dataframes.items():
                for df in df_list:
                    summary['total_rows'] += len(df)
                    summary['total_memory_mb'] += df.memory_usage(deep=True).sum() / (1024 * 1024)
            
            summary['total_memory_mb'] = round(summary['total_memory_mb'], 2)
            
            self.logger.info(
                f"Processing summary: {summary['total_tables']} tables, "
                f"{summary['total_dataframes']} DataFrames, "
                f"{summary['total_rows']} rows, "
                f"{summary['total_memory_mb']} MB"
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating processing summary: {e}")
        
        return summary