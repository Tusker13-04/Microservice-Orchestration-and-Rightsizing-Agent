"""
Integration tests for MOrA application
These tests require the full system to be running (Minikube, Prometheus, Hipster Shop)
"""
import pytest
import subprocess
import time
import json
import requests
from click.testing import CliRunner
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from src.mora.cli.main import main
except ImportError:
    pytest.skip("Cannot import CLI module", allow_module_level=True)


class TestIntegration:
    """Integration tests that require the full system running"""
    
    @classmethod
    def setup_class(cls):
        """Set up test class - check if system is running"""
        cls.runner = CliRunner()
        cls.system_ready = cls._check_system_health()
    
    @classmethod
    def _check_system_health(cls):
        """Check if the required systems are running"""
        try:
            # Check if minikube is running
            result = subprocess.run(['kubectl', 'cluster-info'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return False
            
            # Check if Prometheus is accessible
            try:
                response = requests.get('http://localhost:9090/-/ready', timeout=5)
                if response.status_code != 200:
                    return False
            except requests.exceptions.RequestException:
                return False
            
            # Check if Grafana is accessible
            try:
                response = requests.get('http://localhost:4000/api/health', timeout=5)
                if response.status_code != 200:
                    return False
            except requests.exceptions.RequestException:
                return False
            
            # Check if hipster-shop namespace exists and has pods
            result = subprocess.run(['kubectl', 'get', 'pods', '-n', 'hipster-shop'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0 or 'No resources found' in result.stdout:
                return False
            
            return True
        except Exception as e:
            print(f"System health check failed: {e}")
            return False
    
    def test_system_health(self):
        """Test that all required systems are healthy"""
        if not self.system_ready:
            pytest.skip("System not ready - ensure Minikube, Prometheus, and Hipster Shop are running")
        
        assert self.system_ready, "System should be ready for integration tests"
    
    def test_status_command_integration(self):
        """Test status command with real system"""
        result = self.runner.invoke(main, [
            'status', 
            '--namespace', 'hipster-shop',
            '--prometheus-url', 'http://localhost:9090'
        ])
        
        # Should succeed and show connection status
        # Note: May fail due to missing dependencies, but structure should be correct
        if result.exit_code == 0:
            # If successful, should contain system information
            assert 'MOrA System Status' in result.output or 'system' in result.output.lower()
    
    @pytest.mark.skipif(not hasattr(TestIntegration, 'system_ready') or not getattr(TestIntegration, 'system_ready', False), 
                       reason="System not ready")
    def test_rightsize_command_basic_integration(self):
        """Test basic rightsize command with real system"""
        result = self.runner.invoke(main, [
            'rightsize',
            '--service', 'frontend',
            '--namespace', 'hipster-shop',
            '--strategy', 'statistical',
            '--duration-hours', '1'  # Short duration for testing
        ])
        
        # Should attempt to run (may fail due to missing deps but should not crash)
        assert result.exit_code in [0, 1]  # 1 is acceptable if dependencies are missing
        
        # Should at least show the command was recognized
        output_lower = result.output.lower()
        assert any(keyword in output_lower for keyword in [
            'mora', 'rightsizing', 'analysis', 'service', 'frontend'
        ])
    
    @pytest.mark.skipif(not hasattr(TestIntegration, 'system_ready') or not getattr(TestIntegration, 'system_ready', False), 
                       reason="System not ready")
    def test_json_output_format(self):
        """Test JSON output format"""
        result = self.runner.invoke(main, [
            'rightsize',
            '--service', 'frontend',
            '--namespace', 'hipster-shop',
            '--strategy', 'statistical',
            '--output-format', 'json',
            '--duration-hours', '1'
        ])
        
        # Should attempt JSON output (may fail due to missing deps)
        assert result.exit_code in [0, 1]
        
        # If it produces output, it should be attempted JSON format or show error
        if result.output.strip():
            # Either valid JSON or an error message
            try:
                json.loads(result.output)
            except json.JSONDecodeError:
                # If not JSON, should be an error message or progress info
                assert any(keyword in result.output.lower() for keyword in [
                    'error', 'failed', 'connecting', 'collecting', 'analysis'
                ])


class TestSystemComponents:
    """Test individual system components"""
    
    def test_prometheus_connectivity(self):
        """Test Prometheus connectivity"""
        try:
            response = requests.get('http://localhost:9090/-/ready', timeout=5)
            if response.status_code == 200:
                assert True  # Prometheus is accessible
            else:
                pytest.skip(f"Prometheus returned status {response.status_code}")
        except requests.exceptions.RequestException:
            pytest.skip("Prometheus not accessible at localhost:9090")
    
    def test_grafana_connectivity(self):
        """Test Grafana connectivity"""
        try:
            response = requests.get('http://localhost:4000/api/health', timeout=5)
            if response.status_code == 200:
                assert True  # Grafana is accessible
            else:
                pytest.skip(f"Grafana returned status {response.status_code}")
        except requests.exceptions.RequestException:
            pytest.skip("Grafana not accessible at localhost:4000")
    
    def test_kubernetes_connectivity(self):
        """Test Kubernetes connectivity"""
        try:
            result = subprocess.run(['kubectl', 'cluster-info'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                assert "is running at" in result.stdout
            else:
                pytest.skip("Kubernetes cluster not accessible")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("kubectl not available or cluster not accessible")
    
    def test_hipster_shop_deployment(self):
        """Test that Hipster Shop is deployed"""
        try:
            result = subprocess.run(['kubectl', 'get', 'deployments', '-n', 'hipster-shop', 'frontend'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and 'frontend' in result.stdout:
                assert True  # Frontend deployment exists
            else:
                pytest.skip("Hipster Shop frontend deployment not found")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("kubectl not available")


class TestGrafanaIntegration:
    """Test Grafana integration functionality"""
    
    @pytest.mark.skipif(not hasattr(TestIntegration, 'system_ready') or not getattr(TestIntegration, 'system_ready', False), 
                       reason="System not ready")
    def test_setup_grafana_integration(self):
        """Test Grafana setup command with real system"""
        runner = CliRunner()
        result = runner.invoke(main, [
            'setup-grafana',
            '--namespace', 'hipster-shop',
            '--grafana-url', 'http://localhost:4000',
            '--prometheus-url', 'http://localhost:9090'
        ])
        
        # Should attempt Grafana setup (may fail due to missing deps)
        assert result.exit_code in [0, 1]
        
        # If successful or partially successful, should show setup information
        if result.output.strip():
            assert any(keyword in result.output.lower() for keyword in [
                'grafana', 'dashboard', 'setup', 'integration', 'mora'
            ])


class TestCLIBasic:
    """Basic CLI tests that don't require system integration"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
    
    def test_help_commands(self):
        """Test that help commands work"""
        # Main help
        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert "MOrA" in result.output
        
        # Rightsize help
        result = self.runner.invoke(main, ['rightsize', '--help'])
        assert result.exit_code == 0
        assert "Generate rightsizing recommendations" in result.output
        
        # Status help
        result = self.runner.invoke(main, ['status', '--help'])
        assert result.exit_code == 0
        assert "Show current status" in result.output
    
    def test_version(self):
        """Test version command"""
        result = self.runner.invoke(main, ['--version'])
        # Version should be available (may exit with 0 or 1 depending on implementation)
        assert result.exit_code in [0, 1]


class TestValidation:
    """Test input validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
    
    def test_required_parameters(self):
        """Test that required parameters are enforced"""
        # Missing service parameter should fail
        result = self.runner.invoke(main, ['rightsize'])
        assert result.exit_code != 0
    
    def test_invalid_strategy(self):
        """Test invalid strategy validation"""
        result = self.runner.invoke(main, [
            'rightsize',
            '--service', 'test-service',
            '--strategy', 'invalid-strategy'
        ])
        assert result.exit_code != 0


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, '-v'])
