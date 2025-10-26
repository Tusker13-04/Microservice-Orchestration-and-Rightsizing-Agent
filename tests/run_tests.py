#!/usr/bin/env python3
"""
Test runner script for MOrA application
"""
import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import pytest
        import pandas
        import click
        print("✅ Core testing dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install pytest pandas click")
        return False

def run_unit_tests():
    """Run unit tests"""
    print("\n🧪 Running Unit Tests...")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/test_k8s_client.py',
            'tests/test_statistical_strategy.py', 
            'tests/test_prometheus_client.py',
            'tests/test_data_pipeline.py',
            'tests/test_cli.py',
            '-v', '--tb=short'
        ], cwd=project_root, capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running unit tests: {e}")
        return False

def run_integration_tests():
    """Run integration tests (requires system to be running)"""
    print("\n🔗 Running Integration Tests...")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/test_integration.py',
            '-v', '--tb=short'
        ], cwd=project_root, capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running integration tests: {e}")
        return False

def check_system_health():
    """Check if the system is ready for integration tests"""
    print("\n🔍 Checking System Health...")
    
    # Check kubectl
    try:
        result = subprocess.run(['kubectl', 'cluster-info'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Kubernetes cluster accessible")
        else:
            print("❌ Kubernetes cluster not accessible")
            return False
    except Exception as e:
        print(f"❌ Error checking Kubernetes: {e}")
        return False
    
    # Check Prometheus
    try:
        import requests
        response = requests.get('http://localhost:9090/-/ready', timeout=5)
        if response.status_code == 200:
            print("✅ Prometheus accessible")
        else:
            print("❌ Prometheus not accessible")
            return False
    except Exception as e:
        print(f"❌ Error checking Prometheus: {e}")
        return False
    
    # Check Hipster Shop
    try:
        result = subprocess.run(['kubectl', 'get', 'pods', '-n', 'hipster-shop'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and 'No resources found' not in result.stdout:
            print("✅ Hipster Shop deployed")
        else:
            print("❌ Hipster Shop not found")
            return False
    except Exception as e:
        print(f"❌ Error checking Hipster Shop: {e}")
        return False
    
    return True

def run_simple_cli_test():
    """Run a simple CLI test to verify basic functionality"""
    print("\n🚀 Running Simple CLI Test...")
    try:
        # Test help command
        from src.mora.cli.main import main
        from click.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        
        if result.exit_code == 0 and "MOrA" in result.output:
            print("✅ CLI help command works")
            return True
        else:
            print(f"❌ CLI help command failed: {result.output}")
            return False
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run MOrA tests')
    parser.add_argument('--unit-only', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration-only', action='store_true', help='Run only integration tests')
    parser.add_argument('--cli-only', action='store_true', help='Run only CLI test')
    parser.add_argument('--check-system', action='store_true', help='Check system health only')
    args = parser.parse_args()
    
    print("🧪 MOrA Test Runner")
    print("=" * 50)
    
    if not check_dependencies():
        sys.exit(1)
    
    if args.check_system:
        if check_system_health():
            print("\n✅ System is ready for integration tests!")
        else:
            print("\n❌ System is not ready. Run setup scripts first.")
        return
    
    if args.cli_only:
        run_simple_cli_test()
        return
    
    success = True
    
    if not args.integration_only:
        if not run_unit_tests():
            success = False
            print("❌ Unit tests failed")
    
    if not args.unit_only:
        if check_system_health():
            if not run_integration_tests():
                success = False
                print("❌ Integration tests failed")
        else:
            print("⚠️  Skipping integration tests - system not ready")
            print("   Run setup scripts and try again")
    
    # Always run CLI test if unit tests passed
    if not args.integration_only and not args.unit_only:
        if not run_simple_cli_test():
            success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests completed successfully!")
    else:
        print("❌ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
