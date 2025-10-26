# MOrA Development Guide

## Quick Setup

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 2. Test the CLI

```bash
# Test basic CLI functionality
mora --help
mora rightsize --help
mora status --help
```

### 3. Development Workflow

#### Current Implementation Status
1. **✅ Phase 1 Complete**: Environment setup, monitoring stack, and CLI implementation
2. **✅ Phase 2 Enhanced**: Triple-loop data acquisition with critical architecture fix
3. **🔄 Next Focus**: Phase 3 Orchestrator Agent implementation

#### Recent Critical Enhancements (Phase 2.2)
- **Triple-loop architecture**: Fixed critical flaw in scenario selection
- **48 experiments**: 4 replicas × 6 loads × 2 scenarios (vs. previous 16)
- **Data quality validation**: Comprehensive checks for completeness, NaN values, and stability

#### Testing
```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src/mora tests/
```

#### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Architecture Overview

### Module Structure
- `cli/`: Command-line interface and user interactions
- `core/`: Orchestrator agent and coordination logic
- `models/`: ML model implementations and training
- `monitoring/`: Prometheus and metrics integration
- `k8s/`: Kubernetes client and resource management
- `utils/`: Shared utilities and helpers

### Configuration
- Main config: `config/default.yaml`
- CLI arguments override config file settings
- Environment variables for sensitive data

## Enhanced Features

### Training Data Collection
```bash
# Run enhanced triple-loop training experiments
mora train clean-experiments --service frontend --config-file config/default.yaml

# View training configuration and expected experiment count
# Output: "Total Experiments: 48 (= 2 scenarios × 4 replicas × 6 load levels)"
```

### Data Quality Validation
The system now validates collected data with:
- **Completeness**: ≥90% of required metrics collected
- **NaN values**: ≤5% threshold for critical metrics
- **Stability**: Coefficient of variation ≤10% in last 5 minutes

## Next Steps

1. **Phase 3**: Implement Orchestrator Agent & Predictive Strategy
2. **Phase 4**: Rigorous Evaluation & Paper Formulation
