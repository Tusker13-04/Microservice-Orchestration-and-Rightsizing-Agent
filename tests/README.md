# MOrA Test Suite

This directory contains comprehensive tests for the MOrA (Microservices-Aware Orchestrator Agent) application.

## Test Structure

### Core Test Files

- **`test_cli.py`** - Tests for CLI functionality with mocked dependencies
- **`test_data_acquisition.py`** - **NEW**: Tests for triple-loop architecture and data quality validation
- **`test_k8s_client.py`** - Unit tests for Kubernetes client
- **`test_statistical_strategy.py`** - Unit tests for statistical rightsizing strategy
- **`test_prometheus_client.py`** - Unit tests for Prometheus client
- **`test_data_pipeline.py`** - Unit tests for data pipeline integration
- **`test_grafana_client.py`** - Unit tests for Grafana client integration
- **`test_integration.py`** - Integration tests requiring full system setup

### Test Runner Scripts

- **`run_tests.py`** - Main test runner with various options
- **`run_comprehensive_test.py`** - Runs all tests in sequence
- **`validate_implementation.py`** - Quick validation without dependencies
- **`test_no_deps.py`** - Structure validation without external dependencies
- **`test_system_ready.py`** - System readiness check

## Running Tests

### Quick System Check
```bash
# Check if system is ready
python3 tests/test_system_ready.py

# Quick code validation
python3 tests/test_no_deps.py
```

### Full Test Suite
```bash
# Run comprehensive test suite
python3 tests/run_comprehensive_test.py

# Or run specific test categories
python3 tests/run_tests.py --unit-only
python3 tests/run_tests.py --integration-only
python3 tests/run_tests.py --check-system
```

### Individual Tests
```bash
# Run unit tests with pytest
python3 -m pytest tests/test_cli.py -v

# Run NEW data acquisition tests for triple-loop verification
python3 -m pytest tests/test_data_acquisition.py::TestDataAcquisitionTripleLoop -v

# Run data quality validation tests
python3 -m pytest tests/test_data_acquisition.py::TestDataQualityValidation -v

# Run integration tests (requires system running)
python3 -m pytest tests/test_integration.py -v
```

## Test Categories

### 1. Unit Tests
- **Purpose**: Test individual components in isolation
- **Dependencies**: Minimal (uses mocks)
- **Files**: `test_*.py` (individual component tests)
- **NEW**: Enhanced coverage for triple-loop architecture and data quality validation

### 2. Integration Tests
- **Purpose**: Test full system integration
- **Dependencies**: 
  - Minikube cluster running
  - Prometheus accessible at localhost:9090
  - Hipster Shop deployed in `hipster-shop` namespace
- **Files**: `test_integration.py`

### 3. System Tests
- **Purpose**: Verify system readiness and CLI functionality
- **Dependencies**: Basic system components
- **Files**: `test_system_ready.py`, `validate_implementation.py`

## Prerequisites

### For Unit Tests
```bash
pip install pytest click rich
```

### For Full Testing
```bash
pip install -r requirements.txt
```

### System Requirements for Integration Tests
1. Minikube running with cluster accessible
2. Monitoring stack deployed (Prometheus, Grafana)
3. Hipster Shop application deployed
4. Port forwards active (Prometheus on 9090, Grafana on 4000)

## Test Output

Tests provide colored output with clear success/failure indicators:
- ✅ **PASSED** - Test completed successfully
- ❌ **FAILED** - Test failed (see error details)
- ⚠️ **WARNINGS** - Non-critical issues detected
- 🔍 **INFO** - Test progress and system checks

## Continuous Testing

To run tests as part of development workflow:

```bash
# Quick validation during development
python3 tests/test_no_deps.py

# Full validation before commits
python3 tests/run_comprehensive_test.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Install missing dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. **Connection Failures**: Check system readiness
   ```bash
   python3 tests/test_system_ready.py
   ```

3. **Kubernetes Issues**: Verify cluster status
   ```bash
   kubectl cluster-info
   kubectl get pods -n hipster-shop
   ```

4. **Prometheus Issues**: Check port forward
   ```bash
   curl http://localhost:9090/-/ready
   ```
