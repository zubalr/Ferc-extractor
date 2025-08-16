# FERC EQR Scraper & Processor

A high-performance command-line application for downloading, processing, and storing Electric Quarterly Reports (EQR) data from the Federal Energy Regulatory Commission (FERC).

## Features

- **High-Performance Downloads**: Streaming downloads with progress bars and resume capability
- **XML Processing**: Uses the official `catalystcoop.ferc-xbrl-extractor` library for data extraction
- **Database Storage**: Efficient chunked loading into PostgreSQL or SQLite databases
- **Rich CLI**: Comprehensive command-line interface with detailed logging
- **Error Recovery**: Robust error handling with automatic retry and resume capabilities
- **Memory Efficient**: Optimized for processing multi-gigabyte files without excessive memory usage

## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Install uv

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Install the Application

```bash
# Clone the repository
git clone <repository-url>
cd ferc-eqr-scraper

# Install dependencies
uv sync

# Or install in development mode
uv sync --dev
```

## Quick Start

### Basic Usage

```bash
# Download and process all quarters for 2023
uv run python main.py --start-year 2023 --end-year 2023

# Download specific quarters for multiple years
uv run python main.py --start-year 2022 --end-year 2023 --quarters 1 --quarters 4

# Resume interrupted processing
uv run python main.py --start-year 2023 --end-year 2023 --resume
```

### Configuration

Set environment variables for custom configuration:

```bash
# Database configuration (default: SQLite)
export FERC_DATABASE_URI="postgresql://user:password@localhost:5432/ferc_data"

# Logging level (default: INFO)
export LOG_LEVEL="DEBUG"

# Performance tuning (default: 10000)
export CHUNK_SIZE="5000"

# Network settings (default: 3)
export MAX_RETRIES="5"
```

## Command Line Options

```
Usage: main.py [OPTIONS]

  FERC EQR Scraper & Processor - Download and process FERC Electric Quarterly Reports.

Options:
  --start-year INTEGER            First year to download data for (e.g., 2020)
  --end-year INTEGER              Last year to download data for (e.g., 2023)
  --quarters [1|2|3|4]            Specific quarters to download (can be used multiple times)
  --resume                        Resume interrupted downloads and processing
  --download-only                 Only download files, skip processing and database loading
  --process-only                  Only process existing files, skip downloading
  --verbose, -v                   Enable verbose logging
  --dry-run                       Show what would be done without actually doing it
  --help                          Show this message and exit.
```

## Usage Examples

### Download Only

Download files without processing:

```bash
uv run python main.py --start-year 2023 --end-year 2023 --download-only
```

### Process Existing Files

Process previously downloaded files:

```bash
uv run python main.py --start-year 2023 --end-year 2023 --process-only
```

### Specific Quarters

Download only Q1 and Q4 data:

```bash
uv run python main.py --start-year 2022 --end-year 2023 --quarters 1 --quarters 4
```

### Dry Run

See what would be downloaded without actually doing it:

```bash
uv run python main.py --start-year 2023 --end-year 2023 --dry-run
```

### Verbose Logging

Enable detailed logging for troubleshooting:

```bash
uv run python main.py --start-year 2023 --end-year 2023 --verbose
```

## Database Configuration

### SQLite (Default)

No additional setup required. Data is stored in `ferc_data.db`:

```bash
# Uses default SQLite database
uv run python main.py --start-year 2023 --end-year 2023
```

### PostgreSQL (Recommended for Production)

1. Install PostgreSQL and create a database:

```sql
CREATE DATABASE ferc_data;
CREATE USER ferc_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE ferc_data TO ferc_user;
```

2. Set the database URI:

```bash
export FERC_DATABASE_URI="postgresql://ferc_user:your_password@localhost:5432/ferc_data"
uv run python main.py --start-year 2023 --end-year 2023
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=. --cov-report=html

# Run specific test file
uv run pytest test_main.py -v

# Run integration tests
uv run pytest test_integration.py -v
```

### Code Quality

```bash
# Format code
uv run black .

# Lint code
uv run flake8 .

# Type checking
uv run mypy .
```

### Adding Dependencies

```bash
# Add runtime dependency
uv add package_name

# Add development dependency
uv add --dev package_name

# Update dependencies
uv sync
```

## Architecture

The application is built with a modular architecture:

- **`main.py`**: CLI interface and pipeline orchestration
- **`settings.py`**: Configuration management
- **`downloader.py`**: High-performance file downloading
- **`processor.py`**: XML processing and data extraction
- **`database.py`**: Database operations and schema management
- **`utils.py`**: Shared utilities and helpers

## Performance Tuning

### Memory Usage

- **Chunk Size**: Adjust `CHUNK_SIZE` environment variable (default: 10000 rows)
- **Database Connection Pooling**: Automatically configured for PostgreSQL
- **Streaming Processing**: Files are processed in chunks to minimize memory usage

### Network Performance

- **Concurrent Downloads**: Downloads are optimized with connection reuse
- **Resume Capability**: Interrupted downloads can be resumed
- **Retry Logic**: Automatic retry with exponential backoff

### Database Performance

- **Chunked Loading**: Data is loaded in configurable chunks
- **Transaction Management**: Ensures data consistency
- **Index Optimization**: Tables are created with appropriate data types

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```bash
   # Reduce chunk size
   export CHUNK_SIZE="5000"
   ```

2. **Network Timeouts**
   ```bash
   # Increase timeout and retries
   export REQUEST_TIMEOUT="60"
   export MAX_RETRIES="5"
   ```

3. **Database Connection Issues**
   ```bash
   # Check database URI format
   export FERC_DATABASE_URI="postgresql://user:pass@host:port/database"
   ```

4. **Disk Space Issues**
   - Downloaded files are stored in `ferc_data/downloads/`
   - Temporary files are cleaned up automatically
   - Consider using `--download-only` to download files to external storage

### Logging

Logs are written to both console and files in the `logs/` directory:

```bash
# View recent logs
tail -f logs/ferc_scraper_$(date +%Y%m%d).log

# Enable debug logging
export LOG_LEVEL="DEBUG"
```

### Resume Interrupted Processing

If processing is interrupted, use the `--resume` flag:

```bash
uv run python main.py --start-year 2023 --end-year 2023 --resume
```

## Data Output

### Database Tables

The processed data is stored in tables based on the FERC EQR structure:

- Transaction data (contracts, sales, purchases)
- Company information
- Market data
- Regulatory filings

### Table Schema

Tables are created automatically with appropriate data types. Common columns include:

- `report_date`: Reporting period
- `respondent_id`: Company identifier  
- `contract_id`: Contract identifier
- `transaction_date`: Transaction date
- `quantity`: Energy quantity
- `price`: Transaction price

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `uv run pytest`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the logs in the `logs/` directory
3. Open an issue on GitHub with:
   - Command used
   - Error message
   - Log files (with sensitive information removed)

## Acknowledgments

- [Catalyst Cooperative](https://catalyst.coop/) for the `ferc-xbrl-extractor` library
- [FERC](https://www.ferc.gov/) for providing the EQR data
- The Python community for the excellent libraries used in this project