# Arcanum Testing Guide

This document provides instructions for writing and running tests for the Arcanum codebase.

## Test Structure

The test suite is organized into two main categories:

1. **Unit Tests**: Testing individual components and functions in isolation
2. **Integration Tests**: Testing how components work together and interact with the file system

All tests are located in the `tests/` directory:

```
tests/
├── conftest.py          # Shared pytest fixtures and configuration
├── integration/         # Integration tests
│   ├── test_g3d_tiles.py
│   ├── test_osm.py
│   ├── test_building_processor.py
│   └── test_storage.py
├── test_data/           # Sample data for tests
│   ├── integrated_output/
│   ├── local_server/
│   ├── osm/
│   ├── server/
│   ├── textures/
│   └── unity/
├── unit/                # Unit tests 
│   ├── test_ai.py
│   ├── test_integration.py
│   ├── test_osm.py
│   ├── test_building_processor.py
│   ├── test_storage.py
│   └── test_utilities.py
└── test_*.py            # Legacy tests
```

## Running Tests

You can run tests using the provided `run_tests.py` script, which handles Python path configuration and provides several options.

### Running All Tests

```bash
python run_tests.py
```

### Running Only Unit Tests

```bash
python run_tests.py --unit
```

### Running Only Integration Tests

```bash
python run_tests.py --integration
```

### Running Specific Tests

```bash
# Run a specific test file
python run_tests.py --path tests/unit/test_osm.py

# Run a specific test directory
python run_tests.py --path tests/integration/

# Run tests with specific markers
python run_tests.py --markers "not slow"
```

### Other Options

- `--verbosity` or `-v`: Set verbosity level (0-3)
- `--failfast` or `-f`: Stop after first failure
- `--markers` or `-m`: Pytest markers to select specific tests

## Writing Tests

### Test Dependencies

The test suite uses pytest for running tests. You'll need to install it first:

```bash
pip install pytest
```

### Test Fixtures

Common test fixtures are defined in `tests/conftest.py`, including:

- `temp_dir`: Provides a temporary directory that's cleaned up after the test
- `test_data_dir`: Path to the test data directory
- `sample_bounds`: Sample geographic bounds for testing
- `sample_image_path`: Path to a sample image for testing
- `mock_api_key`: Mock API key for testing external services

### Writing Unit Tests

Unit tests should:
- Test individual functions and classes in isolation
- Mock external dependencies
- Not depend on external services or APIs
- Be fast and deterministic

Example:

```python
def test_my_function(temp_dir, mock_api_key):
    """Test that my_function works correctly."""
    with patch('modules.some_module.external_function') as mock_external:
        mock_external.return_value = "mocked result"
        
        result = my_function(temp_dir, mock_api_key)
        
        assert result["success"] is True
        assert "output_path" in result
        assert os.path.exists(result["output_path"])
```

### Writing Integration Tests

Integration tests should:
- Test how components work together
- Test interactions with the file system
- Mock external services, but use real file I/O
- Validate end-to-end workflows

Example:

```python
def test_end_to_end_process(temp_dir, sample_input_file):
    """Test the end-to-end processing workflow."""
    # Set up test environment
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the process with mocked external API
    with patch('modules.api_client.fetch_data') as mock_fetch:
        mock_fetch.return_value = {"data": "sample response"}
        
        result = run_full_process(sample_input_file, output_dir)
        
        # Verify results
        assert result["success"] is True
        assert len(result["processed_files"]) > 0
        
        # Verify expected files were created
        expected_files = ["output.json", "report.txt"]
        for filename in expected_files:
            assert os.path.exists(os.path.join(output_dir, filename))
```

## Test Coverage

The test suite aims to cover:

1. **Core functionality**: Essential components and workflows
2. **Edge cases**: Error handling and boundary conditions
3. **API compatibility**: Ensuring consistent API contracts
4. **Regression tests**: Preventing previously fixed bugs from returning

## Best Practices

1. **Isolation**: Tests should run independently and not affect each other
2. **Mocking**: Use mocks for external dependencies and services
3. **Fixtures**: Prefer fixtures for common setup and teardown logic
4. **Parameterization**: Use pytest's parameterize decorator for testing multiple cases
5. **Documentation**: Document what each test is verifying and why
6. **Error Handling**: Test both success cases and error cases
7. **Naming**: Use descriptive test names following the pattern `test_<what>_<expected_outcome>`

## Test Data

The `tests/test_data/` directory contains sample data for testing, including:

- OSM building data
- Sample textures
- Mock API responses
- Example configuration files

When adding new tests, try to reuse existing test data or add minimal samples to this directory.

## CI/CD Integration

In the future, the test suite will be integrated with CI/CD pipelines to automatically run tests on code changes, ensuring quality and preventing regressions.