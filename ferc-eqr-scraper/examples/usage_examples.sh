#!/bin/bash
# FERC EQR Scraper Usage Examples

# Basic usage - download and process all quarters for 2023
echo "Example 1: Basic usage for 2023"
uv run python main.py --start-year 2023 --end-year 2023

# Download specific quarters across multiple years
echo "Example 2: Specific quarters for multiple years"
uv run python main.py --start-year 2022 --end-year 2023 --quarters 1 --quarters 4

# Download only (no processing)
echo "Example 3: Download only"
uv run python main.py --start-year 2023 --end-year 2023 --download-only

# Process existing files (no downloading)
echo "Example 4: Process existing files"
uv run python main.py --start-year 2023 --end-year 2023 --process-only

# Resume interrupted processing
echo "Example 5: Resume interrupted processing"
uv run python main.py --start-year 2023 --end-year 2023 --resume

# Dry run to see what would be downloaded
echo "Example 6: Dry run"
uv run python main.py --start-year 2023 --end-year 2023 --dry-run

# Verbose logging for troubleshooting
echo "Example 7: Verbose logging"
uv run python main.py --start-year 2023 --end-year 2023 --verbose

# Single quarter for single year
echo "Example 8: Single quarter"
uv run python main.py --start-year 2023 --end-year 2023 --quarters 1

# Historical data download (multiple years, all quarters)
echo "Example 9: Historical data"
uv run python main.py --start-year 2020 --end-year 2023

# With custom environment variables
echo "Example 10: Custom configuration"
export FERC_DATABASE_URI="postgresql://user:pass@localhost/ferc"
export LOG_LEVEL="DEBUG"
export CHUNK_SIZE="5000"
uv run python main.py --start-year 2023 --end-year 2023