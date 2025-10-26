#!/usr/bin/env python3
"""
Test script to check if the system is ready for full testing
"""
import subprocess
import requests
import sys
from pathlib import Path

def check_kubectl():
    """Check if kubectl is available and cluster is accessible"""
    print("🔍 Checking kubectl...")
    try:
        result = subprocess.run(['kubectl', 'cluster-info'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ kubectl is available and cluster is accessible")
            return True
        else:
            print(f"❌ kubectl cluster-info failed: {result.stderr}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"❌ kubectl not available: {e}")
        return False

def check_prometheus():
    """Check if Prometheus is accessible"""
    print("🔍 Checking Prometheus...")
    try:
        response = requests.get('http://localhost:9090/-/ready', timeout=5)
        if response.status_code == 200:
            print("✅ Prometheus is accessible at localhost:9090")
            return True
        else:
            print(f"❌ Prometheus returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Prometheus not accessible: {e}")
        return False

def check_hipster_shop():
    """Check if Hipster Shop is deployed"""
    print("🔍 Checking Hipster Shop deployment...")
    try:
        result = subprocess.run(['kubectl', 'get', 'pods', '-n', 'hipster-shop', '--no-headers'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            pod_count = len([line for line in result.stdout.strip().split('\n') if line.strip()])
            print(f"✅ Hipster Shop has {pod_count} pods running")
            return True
        else:
            print("❌ No Hipster Shop pods found")
            return False
    except Exception as e:
        print(f"❌ Error checking Hipster Shop: {e}")
        return False

def check_python_deps():
    """Check if Python dependencies are available"""
    print("🔍 Checking Python dependencies...")
    
    required_modules = ['click', 'rich']
    optional_modules = ['pandas', 'kubernetes', 'prometheus_api_client']
    
    missing_required = []
    missing_optional = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} available")
        except ImportError:
            missing_required.append(module)
            print(f"❌ {module} missing (required)")
    
    for module in optional_modules:
        try:
            __import__(module)
            print(f"✅ {module} available")
        except ImportError:
            missing_optional.append(module)
            print(f"⚠️  {module} missing (optional)")
    
    if missing_required:
        print(f"\n❌ Missing required dependencies: {missing_required}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional dependencies: {missing_optional}")
        print("Install with: pip install -r requirements.txt")
    
    return True

def test_basic_cli_structure():
    """Test basic CLI structure without full execution"""
    print("🔍 Checking CLI structure...")
    
    cli_file = Path(__file__).parent.parent / "src/mora/cli/main.py"
    if not cli_file.exists():
        print("❌ CLI main file not found")
        return False
    
    with open(cli_file, 'r') as f:
        content = f.read()
    
    checks = [
        ("main function", "@click.group" in content),
        ("rightsize command", "def rightsize(" in content),
        ("status command", "def status(" in content),
        ("help text", "MOrA" in content and "Microservices" in content),
    ]
    
    for check_name, check_passed in checks:
        if check_passed:
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name}")
            return False
    
    return True

def main():
    """Run system readiness checks"""
    print("🧪 MOrA System Readiness Check")
    print("=" * 50)
    
    checks = [
        ("CLI Structure", test_basic_cli_structure),
        ("Python Dependencies", check_python_deps),
        ("kubectl & Cluster", check_kubectl),
        ("Prometheus", check_prometheus),
        ("Hipster Shop", check_hipster_shop),
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n🔍 {check_name}...")
        try:
            success = check_func()
            results.append((check_name, success))
        except Exception as e:
            print(f"❌ {check_name} failed with exception: {e}")
            results.append((check_name, False))
    
    print("\n" + "=" * 50)
    print("📋 System Readiness Results:")
    
    all_ready = True
    critical_ready = True
    
    for check_name, success in results:
        status = "✅ READY" if success else "❌ NOT READY"
        print(f"  {status}: {check_name}")
        if not success:
            all_ready = False
            # CLI structure and Python deps are critical
            if check_name in ["CLI Structure", "Python Dependencies"]:
                critical_ready = False
    
    if critical_ready:
        print("\n🎉 System is ready for basic testing!")
        if not all_ready:
            print("⚠️  Some optional components are not ready (full integration tests may fail)")
    else:
        print("\n❌ Critical components are not ready. Please check the errors above.")
    
    return 0 if critical_ready else 1

if __name__ == "__main__":
    sys.exit(main())
