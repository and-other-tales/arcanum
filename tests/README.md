# Arcanum Tests

This directory contains tests for the Arcanum codebase, organized into unit tests and integration tests.

## Test Structure

The test suite is organized as follows:

```
tests/
├── conftest.py             # Shared pytest fixtures and configuration
├── integration/            # Integration tests
│   ├── test_ai.py          # Tests for AI module integrations
│   ├── test_storage.py     # Tests for storage module integrations
│   ├── test_g3d_tiles.py   # Tests for Google 3D Tiles integration
│   ├── test_streetview.py  # Tests for Street View integration
│   └── test_workflow.py    # End-to-end workflow tests
├── test_data/              # Test data files
│   ├── images/             # Sample images for testing
│   ├── models/             # Sample models for testing
│   └── fixtures/           # JSON fixtures for mocking
└── unit/                   # Unit tests
    ├── test_ai.py          # Tests for AI module functions
    ├── test_integration.py # Tests for integration module functions
    ├── test_storage.py     # Tests for storage module functions
    ├── test_utilities.py   # Tests for utility functions
    └── test_web.py         # Tests for web module functions
```

## Running Tests

To run the tests, make sure you have pytest installed and run:

```bash
# Run all tests
python -m pytest

# Run unit tests only
python -m pytest tests/unit/

# Run integration tests only
python -m pytest tests/integration/

# Run a specific test file
python -m pytest tests/unit/test_storage.py

# Run tests with coverage
python -m pytest --cov=modules
```

## Test Categories

### Unit Tests

Unit tests test individual functions and classes in isolation, mocking external dependencies as needed. They should be fast and not require external resources.

### Integration Tests

Integration tests test how multiple components work together and may require external resources like file system access. These tests are slower but provide more confidence in the system's behavior.

## Test Data

The `test_data` directory contains data files used by the tests, including sample images, models, and fixtures for mocking external services.

## Contributing Tests

When adding new functionality to Arcanum, please add corresponding tests:

1. Unit tests for individual functions
2. Integration tests for module interactions
3. End-to-end tests for complete workflows

All tests should be independent, idempotent, and should clean up after themselves.