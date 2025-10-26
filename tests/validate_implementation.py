#!/usr/bin/env python3
"""
Quick validation script to test MOrA implementation without full dependencies
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        # Test core imports
        from src.mora.cli.main import main
        print("✅ CLI main module imported")
    except ImportError as e:
        print(f"❌ CLI import failed: {e}")
        return False
    
    try:
        # Test if we can import core components
        try:
            from src.mora.core.data_pipeline import DataPipeline
            print("✅ Data pipeline imported")
        except ImportError as e:
            print(f"⚠️  Data pipeline import failed (may need dependencies): {e}")
        
        try:
            from src.mora.core.statistical_strategy import StatisticalRightsizer
            print("✅ Statistical strategy imported")
        except ImportError as e:
            print(f"❌ Statistical strategy import failed: {e}")
            return False
        
        try:
            from src.mora.k8s.client import KubernetesClient
            print("✅ Kubernetes client imported")
        except ImportError as e:
            print(f"⚠️  Kubernetes client import failed (may need dependencies): {e}")
        
        try:
            from src.mora.monitoring.prometheus_client import PrometheusClient
            print("✅ Prometheus client imported")
        except ImportError as e:
            print(f"⚠️  Prometheus client import failed (may need dependencies): {e}")
            
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    return True

def test_cli_basic():
    """Test basic CLI functionality"""
    print("\n🧪 Testing basic CLI functionality...")
    
    try:
        from click.testing import CliRunner
        from src.mora.cli.main import main
        
        runner = CliRunner()
        
        # Test main help
        result = runner.invoke(main, ['--help'])
        if result.exit_code == 0 and "MOrA" in result.output:
            print("✅ Main help command works")
        else:
            print(f"❌ Main help failed: {result}")
            return False
        
        # Test rightsize help
        result = runner.invoke(main, ['rightsize', '--help'])
        if result.exit_code == 0 and "Generate rightsizing recommendations" in result.output:
            print("✅ Rightsize help command works")
        else:
            print(f"❌ Rightsize help failed: {result}")
            return False
        
        # Test status help
        result = runner.invoke(main, ['status', '--help'])
        if result.exit_code == 0 and "Show current status" in result.output:
            print("✅ Status help command works")
        else:
            print(f"❌ Status help failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False
    
    return True

def test_statistical_strategy():
    """Test statistical strategy functionality"""
    print("\n📊 Testing statistical strategy...")
    
    try:
        from src.mora.core.statistical_strategy import StatisticalRightsizer
        
        # Test initialization
        rightsizer = StatisticalRightsizer(cpu_percentile=95.0, memory_buffer_percentage=15.0)
        assert rightsizer.cpu_percentile == 95.0
        assert rightsizer.memory_buffer_percentage == 15.0
        print("✅ StatisticalRightsizer initialization works")
        
        # Test CPU parsing
        assert rightsizer.parse_cpu_value("100m") == 0.1
        assert rightsizer.parse_cpu_value("1") == 1.0
        print("✅ CPU value parsing works")
        
        # Test memory parsing
        assert rightsizer.parse_memory_value("128Mi") > 0
        assert rightsizer.parse_memory_value("1Gi") > 0
        print("✅ Memory value parsing works")
        
        # Test formatting
        cpu_formatted = rightsizer.format_cpu_value(0.1)
        assert "m" in cpu_formatted or "100" in cpu_formatted
        print("✅ CPU value formatting works")
        
    except Exception as e:
        print(f"❌ Statistical strategy test failed: {e}")
        return False
    
    return True

def check_file_structure():
    """Check that all expected files exist"""
    print("\n📁 Checking file structure...")
    
    expected_files = [
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
        "setup.py",
        "requirements.txt",
        "config/default.yaml"
    ]
    
    missing_files = []
    for file_path in expected_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    return True

def main():
    """Run all validation tests"""
    print("🧪 MOrA Implementation Validation")
    print("=" * 50)
    
    tests = [
        ("File Structure", check_file_structure),
        ("Module Imports", test_imports),
        ("Statistical Strategy", test_statistical_strategy),
        ("CLI Basic", test_cli_basic),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📋 Validation Results:")
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All validations passed! MOrA implementation is ready.")
        return 0
    else:
        print("\n⚠️  Some validations failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
