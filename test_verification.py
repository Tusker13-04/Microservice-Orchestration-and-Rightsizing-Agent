#!/usr/bin/env python3
"""
Quick verification script to test MOrA implementation
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        # Test core imports
        from mora.cli.main import main
        print("✅ CLI main module imported")
    except ImportError as e:
        print(f"❌ CLI import failed: {e}")
        return False
    
    try:
        from mora.core.statistical_strategy import StatisticalRightsizer
        print("✅ Statistical strategy imported")
    except ImportError as e:
        print(f"❌ Statistical strategy import failed: {e}")
        return False
    
    return True

def test_cli_basic():
    """Test basic CLI functionality"""
    print("\n🧪 Testing basic CLI functionality...")
    
    try:
        from click.testing import CliRunner
        from mora.cli.main import main
        
        runner = CliRunner()
        
        # Test main help
        result = runner.invoke(main, ['--help'])
        if result.exit_code == 0 and "MOrA" in result.output:
            print("✅ Main help command works")
            print("CLI help output:", result.output[:200] + "..." if len(result.output) > 200 else result.output)
        else:
            print(f"❌ Main help failed: {result}")
            return False
        
        return True
            
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def test_statistical_strategy():
    """Test statistical strategy functionality"""
    print("\n📊 Testing statistical strategy...")
    
    try:
        from mora.core.statistical_strategy import StatisticalRightsizer
        
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
        
        return True
        
    except Exception as e:
        print(f"❌ Statistical strategy test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🧪 MOrA Implementation Verification")
    print("=" * 50)
    
    tests = [
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
    print("📋 Verification Results:")
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All verifications passed! MOrA implementation is ready.")
        return 0
    else:
        print("\n⚠️  Some verifications failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
