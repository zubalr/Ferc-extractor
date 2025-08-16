# Contributing to FERC EQR Scraper

Thank you for your interest in contributing to the FERC EQR Scraper! This document provides guidelines and information for contributors.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- Git

### Setting up the Development Environment

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ferc-eqr-scraper
   ```

2. **Install dependencies**
   ```bash
   uv sync --dev
   ```

3. **Verify installation**
   ```bash
   uv run pytest
   ```

## Development Workflow

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests**
   ```bash
   # Run all tests
   uv run pytest
   
   # Run with coverage
   uv run pytest --cov=. --cov-report=html
   
   # Run specific test file
   uv run pytest test_main.py -v
   ```

4. **Check code quality**
   ```bash
   # Format code (if you have black installed)
   uv run black .
   
   # Check imports (if you have isort installed)
   uv run isort .
   ```

### Commit Guidelines

- Use clear, descriptive commit messages
- Follow the format: `type: description`
- Types: `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`

Examples:
```
feat: add resume capability for interrupted downloads
fix: handle network timeouts in downloader
docs: update README with PostgreSQL setup instructions
test: add integration tests for database operations
```

## Code Style

### Python Style Guidelines

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Use meaningful variable and function names

### Documentation Style

- Use Google-style docstrings
- Include parameter types and return types
- Provide examples for complex functions

Example:
```python
def download_file(self, url: str, resume: bool = True) -> str:
    """Download a file with progress tracking and resume capability.
    
    Args:
        url: URL to download
        resume: Whether to resume interrupted downloads
        
    Returns:
        Path to the downloaded file
        
    Raises:
        requests.RequestException: If download fails after all retries
        
    Example:
        >>> downloader = FERCDownloader()
        >>> filepath = downloader.download_file("https://example.com/file.zip")
    """
```

## Testing

### Test Structure

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows

### Writing Tests

1. **Test file naming**: `test_<module_name>.py`
2. **Test class naming**: `Test<ClassName>`
3. **Test method naming**: `test_<functionality>_<scenario>`

Example:
```python
class TestFERCDownloader:
    def test_download_file_success(self):
        """Test successful file download."""
        # Test implementation
        
    def test_download_file_network_error(self):
        """Test download with network error."""
        # Test implementation
```

### Test Coverage

- Aim for >90% test coverage
- Test both success and failure scenarios
- Mock external dependencies (network, file system, database)

## Adding New Features

### Before Starting

1. **Check existing issues** to avoid duplicate work
2. **Open an issue** to discuss the feature if it's significant
3. **Review the architecture** to understand how your feature fits

### Feature Development Process

1. **Design the feature**
   - Consider the user interface (CLI options)
   - Plan the implementation approach
   - Identify potential edge cases

2. **Implement the feature**
   - Start with tests (TDD approach recommended)
   - Implement the minimal viable version
   - Add error handling and edge cases

3. **Update documentation**
   - Update README.md if needed
   - Add docstrings to new functions
   - Update CLI help text

4. **Test thoroughly**
   - Unit tests for new functions
   - Integration tests for component interactions
   - Manual testing with real data (if possible)

## Bug Reports

### Before Reporting

1. **Check existing issues** for similar problems
2. **Try the latest version** to see if it's already fixed
3. **Gather information** about your environment

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g., macOS 12.0]
- Python version: [e.g., 3.11.0]
- Package version: [e.g., 1.0.0]

**Additional Context**
- Log files (with sensitive information removed)
- Configuration used
- Any other relevant information
```

## Performance Considerations

### Memory Usage

- Use generators for large data processing
- Implement chunked processing for database operations
- Clean up temporary files promptly
- Monitor memory usage in tests

### Network Efficiency

- Implement connection reuse
- Add appropriate timeouts
- Use streaming for large downloads
- Implement retry logic with backoff

### Database Performance

- Use chunked inserts for large datasets
- Implement proper transaction management
- Consider indexing for frequently queried columns
- Monitor query performance

## Documentation

### Types of Documentation

1. **Code documentation**: Docstrings and comments
2. **User documentation**: README, usage examples
3. **Developer documentation**: Architecture, contributing guidelines
4. **API documentation**: Function and class references

### Documentation Standards

- Keep documentation up-to-date with code changes
- Use clear, concise language
- Provide practical examples
- Include troubleshooting information

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. Update version numbers
2. Update CHANGELOG.md
3. Run full test suite
4. Update documentation
5. Create release notes
6. Tag the release

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Code Reviews**: Pull request discussions

### Questions and Support

- Check the README and documentation first
- Search existing issues and discussions
- Provide detailed information when asking questions
- Be respectful and patient

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain a professional environment

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or inflammatory comments
- Personal attacks
- Spam or off-topic content

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- GitHub contributor statistics

Thank you for contributing to the FERC EQR Scraper!