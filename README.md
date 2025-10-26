# MOrA - Microservices-Aware Orchestrator Agent

A Microservices-Aware Orchestrator Agent for Predictive Kubernetes Rightsizing.

## Overview

MOrA is a research project that addresses the limitations of traditional, single-workload rightsizing tools in microservices environments. It leverages specialized, independently trained ML models and a central orchestrator to provide coordinated, predictive resource recommendations.

## Quick Start

### Prerequisites

- Python 3.9+
- Docker
- Minikube or Kubernetes cluster
- kubectl configured

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd KO

# Install dependencies
pip install -r requirements.txt

# Set up the CLI
pip install -e .
```

### Quick Setup

```bash
# Run the complete setup script (sets up Minikube, Prometheus, and Hipster Shop)
./scripts/setup-minikube.sh

# Install MOrA CLI
pip install -e .
```

### Basic Usage

```bash
# Run baseline statistical rightsizing for frontend service
mora rightsize --strategy statistical --service frontend

# Run predictive rightsizing (after model training)
mora rightsize --strategy predictive --service frontend

# Check system status
mora status --namespace hipster-shop

# Run clean training experiments with enhanced triple-loop architecture
mora train clean-experiments --service frontend --config-file config/default.yaml

# Train ML models for specific services
mora train models --service frontend

# Check trained model status
mora models-status --namespace hipster-shop
```

### Enhanced Training Features

The system now includes a **triple-loop architecture** for data collection:
- **48 total experiments**: 4 replica counts × 6 load levels × 2 scenarios (browsing, checkout)
- **8,640 high-quality data points** for superior ML model training
- **Comprehensive data quality validation** with configurable thresholds

## Project Structure

```
KO/
├── src/
│   ├── mora/
│   │   ├── cli/           # CLI interface
│   │   ├── core/          # Core orchestrator logic
│   │   ├── models/        # ML model implementations
│   │   ├── monitoring/    # Prometheus and monitoring integration
│   │   ├── k8s/          # Kubernetes client integration
│   │   └── utils/        # Utility functions
├── tests/                # Test suite
├── scripts/              # Setup and deployment scripts
├── docs/                 # Documentation
├── config/               # Configuration files
└── data/                 # Training data and model artifacts
```

## Development Phases

See [PRD.md](PRD.md) for detailed project roadmap and phase descriptions.

## Contributing

This is a research project. Please refer to the PRD for the development methodology and phase-specific requirements.

## License

[Add appropriate license]
