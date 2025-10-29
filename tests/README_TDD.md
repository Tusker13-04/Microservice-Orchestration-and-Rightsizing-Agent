# Test-Driven Development (TDD) Guide for MOrA

## Overview

This document outlines the Test-Driven Development workflow for the MOrA project, including test structure, conventions, and best practices.

## What is TDD?

**Test-Driven Development (TDD)** is a software development approach where:
1. **Write tests first** (Red phase - tests fail)
2. **Write code to pass tests** (Green phase - tests pass)
3. **Refactor code** (Refactor phase - improve while keeping tests green)

## Current Test Structure

### Core Test Files

```
tests/
├── test_cli.py                    # CLI command tests (✅ 28 passing)
├── test_no_deps.py                # No-dependency tests (✅ 4 tests)
├── test_system_ready.py          # System readiness tests (✅ 1 test)
├── test_industry_standards_analysis.py  # Industry compliance tests
├── test_model_persistence.py     # Model save/load tests
├── test_professional_ml_pipeline.py  # Professional ML pipeline tests
└── README_TDD.md                 # This file
```

### Test Categories

#### 1. Unit Tests
- **Purpose**: Test individual components in isolation
- **Dependencies**: Minimal (uses mocks)
- **Files**: `test_*.py` (component-specific)
- **Status**: ✅ Partial coverage

#### 2. Integration Tests
- **Purpose**: Test component interactions
- **Dependencies**: Multiple components
- **Files**: `test_cli.py`, `test_integration.py`
- **Status**: ✅ Working

#### 3. End-to-End Tests
- **Purpose**: Test complete workflows
- **Dependencies**: Full system
- **Files**: `test_cli.py`
- **Status**: ✅ Basic coverage

## Running Tests

### Quick Test Run
```bash
# Run all working tests
python3 -m pytest tests/test_cli.py tests/test_no_deps.py tests/test_system_ready.py -v

# Run specific test file
python3 -m pytest tests/test_cli.py -v

# Run specific test class
python3 -m pytest tests/test_cli.py::TestCLI -v

# Run specific test method
python3 -m pytest tests/test_cli.py::TestCLI::test_main_help -v
```

### With Coverage
```bash
# Install coverage tool
pip install pytest-cov

# Run with coverage report
python3 -m pytest tests/ --cov=src --cov-report=html

# View coverage report
xdg-open htmlcov/index.html  # Linux
open htmlcov/index.html      # Mac
```

## TDD Workflow

### Step 1: Red - Write Failing Test
```python
def test_new_feature(self):
    """Test that new feature works correctly"""
    result = new_feature_function()
    assert result.status == "success"
```

### Step 2: Green - Make Test Pass
```python
def new_feature_function():
    # Minimal implementation to pass test
    return {"status": "success"}
```

### Step 3: Refactor - Improve Code
```python
def new_feature_function():
    # Improved implementation while keeping tests green
    # Add error handling, optimizations, etc.
    return {"status": "success"}
```

## Test Naming Conventions

### File Names
- Use `test_` prefix: `test_module_name.py`
- Match module being tested: `test_cli.py` tests `src/mora/cli/`

### Test Classes
- Use `Test` prefix: `class TestModuleName:`
- Group related tests together

### Test Methods
- Use descriptive names: `test_function_name_scenario`
- Examples:
  - `test_train_lightweight_with_service()`
  - `test_evaluate_with_missing_model()`
  - `test_rightsize_with_invalid_service()`

## Test Structure Template

```python
import pytest
from unittest.mock import Mock, patch

class TestModuleName:
    """Test cases for ModuleName"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Initialize test data, mocks, etc.
        pass
    
    def teardown_method(self):
        """Clean up test fixtures"""
        # Clean up after each test
        pass
    
    def test_basic_functionality(self):
        """Test basic functionality works"""
        # Arrange
        test_input = "test"
        
        # Act
        result = function_under_test(test_input)
        
        # Assert
        assert result == expected_output
    
    def test_error_handling(self):
        """Test error handling"""
        with pytest.raises(ValueError):
            function_under_test(invalid_input)
```

## Common Test Patterns

### 1. Testing CLI Commands
```python
def test_command_execution(self):
    """Test CLI command execution"""
    from click.testing import CliRunner
    from src.mora.cli.main import main
    
    runner = CliRunner()
    result = runner.invoke(main, ['command', '--options'])
    
    assert result.exit_code == 0
    assert "expected output" in result.output
```

### 2. Testing with Mocks
```python
@patch('module.function')
def test_with_mock(self, mock_function):
    """Test with mocked dependencies"""
    mock_function.return_value = {"status": "success"}
    
    result = function_under_test()
    
    assert result == expected_output
    mock_function.assert_called_once()
```

### 3. Testing File Operations
```python
def test_file_operations(self):
    """Test file save/load operations"""
    import tempfile
    import os
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test file operations
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, 'w') as f:
            f.write("test")
        
        assert os.path.exists(file_path)
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
```

## Current Test Status

### ✅ Passing Tests (28 tests)
- `test_cli.py` - 23 tests passing
- `test_no_deps.py` - 4 tests passing
- `test_system_ready.py` - 1 test passing

### 🚧 In Development
- Enhanced lightweight pipeline tests
- Unified evaluator tests
- Industry standards analysis tests
- Model persistence tests

### 📋 Test Coverage

| Component | Coverage | Status |
|-----------|----------|--------|
| CLI Commands | ✅ Good | 23 tests |
| Data Collection | ✅ Good | 4 tests |
| Model Training | 🚧 Partial | Needs expansion |
| Model Evaluation | 🚧 Partial | Needs expansion |
| Rightsizing | ✅ Good | Integrated |

## Best Practices

### 1. Write Tests First
- Always write tests before implementation
- Tests document expected behavior
- Tests catch bugs early

### 2. Keep Tests Independent
- Each test should run in isolation
- Use `setup_method` and `teardown_method` for fixtures
- Don't rely on test execution order

### 3. Test Edge Cases
- Test with empty inputs
- Test with invalid inputs
- Test boundary conditions
- Test error handling

### 4. Use Descriptive Assertions
```python
# Bad
assert result == expected

# Good
assert result == expected, f"Expected {expected}, got {result}"
```

### 5. Clean Up After Tests
- Always clean up temporary files
- Reset global state
- Use context managers when possible

## Continuous Integration

### GitHub Actions (Future)
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: pytest tests/
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure project structure is correct
   export PYTHONPATH="${PYTHONPATH}:${PWD}"
   ```

2. **Missing Dependencies**
   ```bash
   pip install pytest pytest-mock pytest-cov
   ```

3. **Broken Tests**
   - Check for API changes
   - Verify mock return values
   - Check file paths and permissions

## Future Improvements

- [ ] Add tests for all CLI commands
- [ ] Enhance model training tests
- [ ] Add performance benchmarks
- [ ] Implement coverage reporting in CI
- [ ] Add mutation testing
- [ ] Create test data generators

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [TDD Best Practices](https://www.agilealliance.org/glossary/tdd/)
- [Testing Best Practices](https://testing.googleblog.com/)

## Contributing

When contributing to MOrA:
1. Write tests first (Red)
2. Implement feature (Green)
3. Refactor if needed (Refactor)
4. Ensure all tests pass
5. Submit pull request

---

**Last Updated**: October 2024
**Test Suite Version**: 1.0
**Passing Tests**: 28/28 ✅

