#!/bin/bash
# Repository Cleanup Script
# Maintains clean repository structure according to industry standards

echo "🧹 MOrA Repository Cleanup Script"
echo "================================="

# Function to check if file exists and remove it
cleanup_file() {
    if [ -f "$1" ]; then
        echo "Removing: $1"
        rm -f "$1"
    fi
}

# Function to check if directory exists and remove it
cleanup_dir() {
    if [ -d "$1" ]; then
        echo "Removing directory: $1"
        rm -rf "$1"
    fi
}

echo "📁 Cleaning up temporary files..."

# Remove temporary Python files
cleanup_file "*.pyc"
cleanup_file "*.pyo"
cleanup_file "__pycache__"
cleanup_file "*.log"
cleanup_file "*.tmp"
cleanup_file "*.temp"

# Remove backup directories
cleanup_dir "backup_*"
cleanup_dir "*_backup"
cleanup_dir "temp_*"
cleanup_dir "tmp_*"

# Remove old configuration files
cleanup_file "prometheus-config.yaml"
cleanup_file "prometheus-patch.yaml"
cleanup_file "*.bak"
cleanup_file "*.old"

# Remove test output files
cleanup_file "test_output.txt"
cleanup_file "*.test"
cleanup_file "debug_*"

# Remove old model files (keep only the latest)
cleanup_file "models/*_old.joblib"
cleanup_file "models/*_backup.joblib"
cleanup_file "models/*_temp.joblib"

echo "📊 Cleaning up Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

echo "🗂️ Organizing files..."

# Ensure proper directory structure exists
mkdir -p train_models evaluate_models utils scripts docs

# Move any stray training scripts
if [ -f "train_*.py" ]; then
    echo "Moving stray training scripts to train_models/"
    mv train_*.py train_models/ 2>/dev/null || true
fi

# Move any stray evaluation scripts
if [ -f "evaluate_*.py" ]; then
    echo "Moving stray evaluation scripts to evaluate_models/"
    mv evaluate_*.py evaluate_models/ 2>/dev/null || true
fi

# Move any stray utility scripts
if [ -f "test_*.py" ] && [ ! -f "test_*.py" ]; then
    echo "Moving stray test scripts to utils/"
    mv test_*.py utils/ 2>/dev/null || true
fi

echo "📋 Repository Status:"
echo "====================="
echo "📁 Directories:"
ls -la | grep "^d" | awk '{print "  " $9}'

echo ""
echo "📄 Key Files:"
echo "  - Models: $(ls models/*.joblib 2>/dev/null | wc -l) trained models"
echo "  - Training Scripts: $(ls train_models/*.py 2>/dev/null | wc -l) scripts"
echo "  - Evaluation Scripts: $(ls evaluate_models/*.py 2>/dev/null | wc -l) scripts"
echo "  - Training Data: $(ls training_data/*.csv 2>/dev/null | wc -l) CSV files"

echo ""
echo "✅ Repository cleanup completed!"
echo "🎯 Repository is now organized according to industry standards"

# Show current structure
echo ""
echo "📊 Current Repository Structure:"
echo "================================"
tree -I 'venv|__pycache__|*.pyc|*.pyo|.git' -L 3 2>/dev/null || {
    echo "  📁 src/"
    echo "  📁 models/"
    echo "  📁 training_data/"
    echo "  📁 train_models/"
    echo "  📁 evaluate_models/"
    echo "  📁 utils/"
    echo "  📁 scripts/"
    echo "  📁 config/"
    echo "  📁 tests/"
    echo "  📁 docs/"
    echo "  📄 README.md"
    echo "  📄 PRD.md"
    echo "  📄 ML-Pipeline.md"
}
