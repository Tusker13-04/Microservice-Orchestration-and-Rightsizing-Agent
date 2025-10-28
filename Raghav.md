# MOrA System Setup Guide for Raghav

## 🎯 Overview

This guide will help you clone, set up, and use the MOrA (Microservice Orchestration and Rightsizing Agent) system on a fresh system with GPU support. The system includes pre-trained models and comprehensive ML pipelines for microservice resource optimization.

## 📋 System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended for professional pipeline)
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: At least 5GB free space
- **CPU**: Multi-core processor (4+ cores recommended)

### Software Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Python**: 3.8 or higher
- **CUDA**: 11.0+ (for GPU support)
- **Git**: For cloning the repository

## 🚀 Quick Start

### Step 1: Clone the Repository
```bash
# Clone the MOrA repository
git clone <repository-url>
cd MOrA

# Verify the repository structure
ls -la
```

**Expected Structure:**
```
MOrA/
├── src/                    # Source code
├── train_models/          # ML training pipelines
├── evaluate_models/       # Evaluation tools
├── evaluation_reports/    # Generated reports
├── models/                # Pre-trained models (5 models, 13MB)
├── training_data/         # Training datasets (8.8MB)
├── config/                # Configuration files
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
└── README.md             # Project overview
```

### Step 2: Install Dependencies

#### Python Environment Setup
```bash
# Create virtual environment (recommended)
python3 -m venv mora_env
source mora_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

#### GPU Setup (for Professional Pipeline)
```bash
# Install CUDA toolkit (if not already installed)
# Ubuntu/Debian:
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Verify CUDA installation
nvcc --version
nvidia-smi

# Install GPU-enabled TensorFlow
pip install tensorflow-gpu
```

### Step 3: Verify Installation
```bash
# Test Python imports
python3 -c "
import tensorflow as tf
import pandas as pd
import numpy as np
from prophet import Prophet
print('✅ All core dependencies imported successfully')
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')
"
```

## 📚 Understanding the System

Before proceeding, please read the following documentation to understand the system architecture:

### Essential Documentation
1. **[README.md](README.md)** - Project overview and quick start
2. **[docs/Setup.md](docs/Setup.md)** - Detailed installation and configuration
3. **[docs/User-Guide.md](docs/User-Guide.md)** - CLI commands and usage examples
4. **[docs/Architecture.md](docs/Architecture.md)** - System architecture and design
5. **[ML-Pipeline.md](ML-Pipeline.md)** - Machine learning pipeline documentation
6. **[PRD.md](PRD.md)** - Product requirements and current status

### Key Concepts
- **Dual Pipeline Architecture**: Lightweight (CPU-friendly) + Professional (GPU-optimized)
- **Pre-trained Models**: 5 services already trained and ready for use
- **Evaluation System**: Comprehensive model assessment with industry standards
- **CLI Interface**: Easy-to-use command-line tools

## 🧪 Testing the System

### Step 1: Basic System Test
```bash
# Test CLI help
python3 -m src.mora.cli.main --help

# Test train commands
python3 -m src.mora.cli.main train --help

# Check system status
python3 -m src.mora.cli.main status
```

### Step 2: Verify Pre-trained Models
```bash
# Check available models
ls -lh models/

# Expected output: 5 models (13MB total)
# - adservice_lstm_prophet_pipeline.joblib
# - cartservice_lstm_prophet_pipeline.joblib
# - checkoutservice_lstm_prophet_pipeline.joblib
# - frontend_lstm_prophet_pipeline.joblib
# - paymentservice_lstm_prophet_pipeline.joblib
```

### Step 3: Test Model Evaluation
```bash
# Evaluate a pre-trained model
python3 -m src.mora.cli.main train evaluate --service frontend

# Evaluate all models
python3 -m src.mora.cli.main train evaluate --all

# Run industry standards analysis
python3 evaluate_models/industry_standards_analysis.py
```

### Step 4: Test Training Pipeline (Lightweight)
```bash
# Test lightweight pipeline (CPU-friendly, 2-3 minutes)
python3 -m src.mora.cli.main train lightweight --service frontend --verbose
```

## 🚀 Professional Pipeline Training (GPU-Optimized)

Since you have GPU support, you can use the professional pipeline for maximum accuracy:

### Step 1: Verify GPU Availability
```bash
# Check GPU status
python3 -c "
import tensorflow as tf
print('GPU devices:', tf.config.list_physical_devices('GPU'))
print('CUDA available:', tf.test.is_built_with_cuda())
"
```

### Step 2: Train Professional Models
```bash
# Train single service with professional pipeline (5 algorithms)
python3 -m src.mora.cli.main train models --service frontend --verbose

# Train multiple services
python3 -m src.mora.cli.main train models --service frontend,cartservice,checkoutservice

# Expected training time: 10-15 minutes per service
# Algorithms used: LSTM, Prophet, XGBoost, LightGBM, RandomForest
```

### Step 3: Monitor Training Progress
```bash
# Check training progress
python3 -m src.mora.cli.main train status --service frontend

# Monitor GPU usage
nvidia-smi -l 1
```

### Step 4: Evaluate Professional Models
```bash
# Evaluate the newly trained professional models
python3 -m src.mora.cli.main train evaluate --service frontend

# Compare with lightweight models
python3 evaluate_models/industry_standards_analysis.py
```

## 📊 Understanding the Results

### Model Performance Metrics
- **MSE (Mean Squared Error)**: Lower is better
- **MAE (Mean Absolute Error)**: Lower is better
- **R² (R-squared)**: Higher is better (closer to 1.0)
- **Confidence**: Model prediction confidence (0.0-1.0)

### Industry Standards Compliance
- **CPU Prediction**: Target < 0.1 MSE, > 0.8 R²
- **Memory Prediction**: Target < 0.2 MSE, > 0.7 R²
- **Replica Prediction**: Target < 0.3 MSE, > 0.6 R²

### Expected Performance
- **Lightweight Pipeline**: 55-60% industry compliance
- **Professional Pipeline**: 70-80% industry compliance (with GPU)

## 🔧 Configuration

### Professional Pipeline Configuration
The professional pipeline uses `config/professional_ml_config.json`:

```json
{
  "lstm": {
    "sequence_length": 10,
    "hidden_units": [32, 16],
    "epochs": 20,
    "batch_size": 32
  },
  "xgboost": {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1
  }
}
```

### GPU Optimization Settings
```bash
# Set TensorFlow GPU memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0
```

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA/GPU Issues
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Check TensorFlow GPU support
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If no GPU detected, install GPU-enabled TensorFlow
pip uninstall tensorflow
pip install tensorflow-gpu
```

#### 2. Memory Issues
```bash
# Reduce batch size in professional pipeline config
# Edit config/professional_ml_config.json
{
  "lstm": {
    "batch_size": 16  # Reduce from 32
  }
}
```

#### 3. Training Data Issues
```bash
# Check training data availability
ls -lh training_data/

# Verify data format
head -5 training_data/frontend_*.csv
```

#### 4. Model Loading Issues
```bash
# Check model files
ls -lh models/

# Test model loading
python3 -c "
import joblib
model = joblib.load('models/frontend_lstm_prophet_pipeline.joblib')
print('Model loaded successfully')
"
```

### Performance Optimization

#### For GPU Training
```bash
# Use mixed precision training
export TF_ENABLE_MIXED_PRECISION=1

# Optimize GPU memory
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

#### For CPU Training (if GPU unavailable)
```bash
# Use lightweight pipeline instead
python3 -m src.mora.cli.main train lightweight --service frontend
```

## 📈 Next Steps

### 1. Experiment with Different Services
```bash
# Train models for different services
python3 -m src.mora.cli.main train models --service adservice
python3 -m src.mora.cli.main train models --service cartservice
```

### 2. Customize Configuration
- Edit `config/professional_ml_config.json` for different hyperparameters
- Adjust GPU memory settings based on your hardware
- Modify evaluation thresholds in `evaluate_models/`

### 3. Analyze Results
- Review evaluation reports in `evaluation_reports/`
- Compare lightweight vs professional pipeline performance
- Analyze industry standards compliance

### 4. Extend the System
- Add new services to the training pipeline
- Implement custom evaluation metrics
- Integrate with your own microservices

## 📞 Support

### Documentation References
- **[docs/User-Guide.md](docs/User-Guide.md)** - Complete CLI reference
- **[docs/API-Reference.md](docs/API-Reference.md)** - API documentation
- **[ML-Pipeline.md](ML-Pipeline.md)** - Detailed ML pipeline info

### Getting Help
1. Check the troubleshooting section above
2. Review the comprehensive documentation in `docs/`
3. Check the evaluation reports for model performance insights
4. Use verbose mode (`--verbose`) for detailed output

## 🎯 Success Criteria

You'll know the system is working correctly when:
- ✅ All CLI commands execute without errors
- ✅ Pre-trained models load and evaluate successfully
- ✅ Professional pipeline trains models in 10-15 minutes
- ✅ GPU is utilized during training (check with `nvidia-smi`)
- ✅ Evaluation reports show industry-standard performance
- ✅ New models achieve 70%+ industry compliance

## 🚀 Ready to Go!

You now have everything needed to use the MOrA system effectively. Start with the basic tests, then move to professional pipeline training with your GPU. The system is production-ready and will provide excellent results for microservice resource optimization.

**Happy Training! 🎉**
