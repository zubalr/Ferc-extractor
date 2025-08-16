# Dockerfile for FERC EQR Scraper
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY *.py ./
COPY examples/ ./examples/

# Install Python dependencies
RUN uv sync --frozen

# Create directories for data and logs
RUN mkdir -p ferc_data/downloads ferc_data/extracted logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["uv", "run", "python", "main.py", "--help"]