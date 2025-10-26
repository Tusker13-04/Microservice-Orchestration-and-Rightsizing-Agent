#!/usr/bin/env python3
"""
Comprehensive test runner for MOrA application
This script tests the implementation at multiple levels
"""
import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_test(test_name, test_script, description):
    """Run a test script and report results"""
    print(f"\n🧪 {test_name}")
    print("=" * 50)
    print(f"📝 {description}")
    
    try:
        result = subprocess.run([sys.executable, test_script], 
                              cwd=project_root, 
                              capture_output=True, 
                              text=True, 
                              timeout=60)
        
        if result.returncode == 0:
            print("✅ PASSED")
            if result.stdout.strip():
                # Show summary of output
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:  # Show last 5 lines
                    if line.strip():
                        print(f"   {line}")
        else:
            print("❌ FAILED")
            if result.stderr:
                print("Error output:")
                print(result.stderr[:500])  # First 500 chars
            if result.stdout and "❌" in result.stdout:
                print("Test output:")
                print(result.stdout[:500])
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT - Test took too long")
        return False
    except Exception as e:
        print(f"💥 ERROR - Test execution failed: {e}")
        return False

def main():
    """Run all tests in sequence"""
    print("🧪 MOrA Comprehensive Test Suite")
    print("=" * 70)
    print("This will run tests at multiple levels to verify the implementation")
    
    # Define test sequence
    tests = [
        ("Code Structure Test", "tests/test_no_deps.py", 
         "Validates file syntax, imports, and code structure without external dependencies"),
        
        ("System Readiness Test", "tests/test_system_ready.py",
         "Checks if the system (Minikube, Prometheus, Hipster Shop) is ready for testing"),
    ]
    
    # Check if we can run integration tests by looking for pytest
    try:
        import pytest
        tests.extend([
            ("Unit Tests", "tests/test_cli.py", 
             "Tests CLI functionality with mocked dependencies"),
            ("Integration Tests", "tests/test_integration.py",
             "Tests with real system (requires all components running)"),
        ])
    except ImportError:
        print("⚠️  pytest not available - skipping unit and integration tests")
    
    # Run tests
    results = []
    for test_name, test_script, description in tests:
        if Path(project_root / test_script).exists():
            success = run_test(test_name, test_script, description)
            results.append((test_name, success))
        else:
            print(f"\n⚠️  {test_name}: Test script {test_script} not found")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 COMPREHENSIVE TEST RESULTS")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:<12} {test_name}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! MOrA implementation is verified and ready.")
        print("\nNext steps:")
        print("1. Install full dependencies: pip install -r requirements.txt")
        print("2. Run the CLI: python3 -m src.mora.cli.main --help")
        print("3. Test with real data: python3 -m src.mora.cli.main status")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
