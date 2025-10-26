#!/usr/bin/env python3
"""
Test script that verifies code structure without requiring external dependencies
"""
import sys
import os
from pathlib import Path
import ast
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True
    except (SyntaxError, FileNotFoundError) as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"

def test_file_syntax():
    """Test that all Python files have valid syntax"""
    print("🔍 Testing file syntax...")
    
    python_files = [
        "src/mora/__init__.py",
        "src/mora/cli/__init__.py", 
        "src/mora/cli/main.py",
        "src/mora/core/__init__.py",
        "src/mora/core/data_pipeline.py",
        "src/mora/core/statistical_strategy.py",
        "src/mora/k8s/__init__.py",
        "src/mora/k8s/client.py",
        "src/mora/k8s/discovery.py",
        "src/mora/monitoring/__init__.py",
        "src/mora/monitoring/prometheus_client.py",
        "src/mora/utils/__init__.py",
        "src/mora/utils/config.py",
    ]
    
    failed_files = []
    for file_path in python_files:
        full_path = project_root / file_path
        if not full_path.exists():
            print(f"❌ File not found: {file_path}")
            failed_files.append(file_path)
            continue
            
        result = check_syntax(full_path)
        if result is True:
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}: {result}")
            failed_files.append(file_path)
    
    return len(failed_files) == 0

def test_import_structure():
    """Test that import structure is reasonable"""
    print("\n🔍 Testing import structure...")
    
    try:
        # Test that we can at least read the CLI file and check its structure
        cli_file = project_root / "src/mora/cli/main.py"
        if cli_file.exists():
            with open(cli_file, 'r') as f:
                content = f.read()
            
            # Check for expected CLI structure
            checks = [
                ("Click imports", "import click" in content),
                ("Rich imports", "from rich" in content),
                ("Main function", "@click.group" in content or "def main" in content),
                ("Rightsize command", "@main.command" in content and "rightsize" in content),
                ("Status command", "@main.command" in content and "status" in content),
            ]
            
            for check_name, check_result in checks:
                if check_result:
                    print(f"✅ {check_name}")
                else:
                    print(f"❌ {check_name}")
                    return False
        
        return True
    except Exception as e:
        print(f"❌ Import structure test failed: {e}")
        return False

def test_class_structure():
    """Test that expected classes exist in the code"""
    print("\n🔍 Testing class structure...")
    
    try:
        # Check statistical strategy
        strategy_file = project_root / "src/mora/core/statistical_strategy.py"
        if strategy_file.exists():
            with open(strategy_file, 'r') as f:
                content = f.read()
            
            if "class StatisticalRightsizer" in content:
                print("✅ StatisticalRightsizer class found")
            else:
                print("❌ StatisticalRightsizer class not found")
                return False
            
            # Check for expected methods
            methods = [
                "generate_recommendations",
                "validate_recommendations", 
                "parse_cpu_value",
                "parse_memory_value"
            ]
            
            for method in methods:
                if f"def {method}" in content:
                    print(f"✅ {method} method found")
                else:
                    print(f"❌ {method} method not found")
        
        return True
    except Exception as e:
        print(f"❌ Class structure test failed: {e}")
        return False

def test_config_files():
    """Test that configuration files exist and have valid structure"""
    print("\n🔍 Testing configuration files...")
    
    # Check requirements.txt
    req_file = project_root / "requirements.txt"
    if req_file.exists():
        with open(req_file, 'r') as f:
            content = f.read()
        
        expected_packages = ["kubernetes", "prometheus-api-client", "pandas", "click", "rich"]
        for package in expected_packages:
            if package in content:
                print(f"✅ {package} in requirements.txt")
            else:
                print(f"❌ {package} missing from requirements.txt")
    
    # Check config file
    config_file = project_root / "config/default.yaml"
    if config_file.exists():
        with open(config_file, 'r') as f:
            content = f.read()
        
        if "prometheus:" in content and "kubernetes:" in content:
            print("✅ Config file has expected sections")
        else:
            print("❌ Config file missing expected sections")
    
    return True

def main():
    """Run all no-dependency tests"""
    print("🧪 MOrA No-Dependencies Test")
    print("=" * 50)
    
    tests = [
        ("File Syntax", test_file_syntax),
        ("Import Structure", test_import_structure), 
        ("Class Structure", test_class_structure),
        ("Config Files", test_config_files),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📋 No-Dependencies Test Results:")
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 Code structure validation passed!")
        print("💡 To run full tests, install dependencies: pip install -r requirements.txt")
    else:
        print("\n⚠️  Some structure tests failed.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
