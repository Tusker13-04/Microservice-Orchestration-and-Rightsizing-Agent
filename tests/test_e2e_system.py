"""
End-to-End System Tests for MOrA

This module contains comprehensive tests that verify the entire MOrA system
from data collection to model training, including:
- System readiness checks
- Prometheus connectivity and metrics collection
- Load generation with JMeter
- Parallel experiment execution
- Data persistence and resumability
- CLI functionality
"""

import pytest
import time
import os
import json
import yaml
import subprocess
import requests
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.mora.monitoring.prometheus_client import PrometheusClient
from src.mora.core.data_acquisition import DataAcquisitionPipeline
from src.mora.core.load_generator import LoadGenerator
from src.mora.cli.main import main
from click.testing import CliRunner


class TestSystemReadiness:
    """Test that all system components are ready"""
    
    def test_minikube_running(self):
        """Verify Minikube is running"""
        result = subprocess.run(['kubectl', 'cluster-info'], 
                                capture_output=True, text=True)
        assert result.returncode == 0, "Minikube cluster not accessible"
        assert "minikube" in result.stdout.lower(), "Not connected to Minikube"
    
    def test_prometheus_accessible(self):
        """Verify Prometheus is accessible"""
        try:
            response = requests.get('http://localhost:9090/-/ready', timeout=5)
            assert response.status_code == 200, "Prometheus not ready"
            assert "ready" in response.text.lower(), "Prometheus not responding correctly"
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Prometheus not accessible: {e}")
    
    def test_hipster_shop_deployed(self):
        """Verify Hipster Shop services are running"""
        result = subprocess.run(['kubectl', 'get', 'pods', '-n', 'hipster-shop'], 
                              capture_output=True, text=True)
        assert result.returncode == 0, "Cannot access hipster-shop namespace"
        
        # Check for key services
        services = ['frontend', 'checkoutservice', 'cartservice']
        for service in services:
            assert service in result.stdout, f"Service {service} not found in hipster-shop"
    
    def test_prometheus_client_connection(self):
        """Test Prometheus client connectivity"""
        client = PrometheusClient('http://localhost:9090')
        assert client.test_connection(), "Prometheus client connection failed"


class TestMetricsCollection:
    """Test comprehensive metrics collection"""
    
    def test_basic_metrics_collection(self):
        """Test that basic system metrics can be collected"""
        client = PrometheusClient('http://localhost:9090')
        
        # Test individual metric collection
        metrics_to_test = [
            'cpu_cores',
            'mem_bytes', 
            'replica_count',
            'node_cpu_util',
            'node_mem_util'
        ]
        
        for metric in metrics_to_test:
            try:
                if metric == 'cpu_cores':
                    result = client._get_cpu_cores('pod=~"frontend.*"', 
                                                client._get_time_range(1))
                elif metric == 'mem_bytes':
                    result = client._get_memory_working_set('pod=~"frontend.*"', 
                                                          client._get_time_range(1))
                elif metric == 'replica_count':
                    result = client._get_replica_count('frontend', 'hipster-shop', 
                                                     client._get_time_range(1))
                elif metric == 'node_cpu_util':
                    result = client._get_node_cpu_utilization(client._get_time_range(1))
                elif metric == 'node_mem_util':
                    result = client._get_node_memory_utilization(client._get_time_range(1))
                
                assert result is not None, f"Metric {metric} returned None"
                # Don't require data points since we're not generating load yet
                
            except Exception as e:
                pytest.fail(f"Failed to collect metric {metric}: {e}")
    
    def test_comprehensive_metrics_structure(self):
        """Test that comprehensive metrics returns expected structure"""
        client = PrometheusClient('http://localhost:9090')
        
        metrics = client.get_comprehensive_metrics('hipster-shop', 'frontend', 1)
        
        # Verify all expected metrics are present
        expected_metrics = [
            'cpu_cores', 'mem_bytes', 'net_rx_bytes', 'net_tx_bytes',
            'throttled_seconds', 'pod_restarts', 'replica_count',
            'node_cpu_util', 'node_mem_util', 'requests_per_second',
            'latency_p50', 'latency_p95', 'latency_p99', 'error_rate'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], type(metrics['cpu_cores'])), \
                f"Metric {metric} has wrong type"


class TestLoadGeneration:
    """Test JMeter load generation functionality"""
    
    def test_jmeter_script_creation(self):
        """Test that JMeter scripts can be created"""
        generator = LoadGenerator()
        
        script_path = generator.create_jmeter_script(
            'test_script', 'localhost', 80, 'browsing', 10
        )
        
        assert os.path.exists(script_path), "JMeter script not created"
        assert script_path.endswith('.jmx'), "Script should have .jmx extension"
        
        # Verify script content
        with open(script_path, 'r') as f:
            content = f.read()
            assert 'ThreadGroup' in content, "Script should contain ThreadGroup"
            assert 'HTTPSampler' in content, "Script should contain HTTPSampler"
    
    def test_jmeter_execution_short(self):
        """Test short JMeter execution (30 seconds)"""
        generator = LoadGenerator()
        
        script_path = generator.create_jmeter_script(
            'test_short', 'localhost', 80, 'browsing', 5
        )
        
        # Run a very short test
        result = generator.run_load_test(script_path, duration_minutes=0.5)
        
        assert result is not None, "Load test should return result"
        # Clean up
        if os.path.exists(script_path):
            os.remove(script_path)


class TestDataAcquisition:
    """Test data acquisition pipeline"""
    
    def test_pipeline_initialization(self):
        """Test that data acquisition pipeline initializes correctly"""
        with open('config/resource-optimized.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        pipeline = DataAcquisitionPipeline(config)
        assert pipeline is not None, "Pipeline should initialize"
        assert hasattr(pipeline, 'run_parallel_training_experiments'), \
            "Pipeline should have parallel execution method"
    
    def test_experiment_id_generation(self):
        """Test experiment ID generation"""
        with open('config/resource-optimized.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        pipeline = DataAcquisitionPipeline(config)
        
        exp_id = pipeline._get_experiment_id('frontend', 'browsing', 1, 10)
        expected = 'frontend_browsing_replicas_1_users_10'
        assert exp_id == expected, f"Expected {expected}, got {exp_id}"
    
    def test_experiment_completion_tracking(self):
        """Test experiment completion tracking"""
        with open('config/resource-optimized.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        pipeline = DataAcquisitionPipeline(config)
        
        # Test non-existent experiment
        exp_id = 'test_nonexistent_experiment'
        assert not pipeline._is_experiment_completed(exp_id), \
            "Non-existent experiment should not be marked as completed"
        
        # Test completed experiments list
        completed = pipeline._get_completed_experiments('frontend')
        assert isinstance(completed, list), "Should return list of completed experiments"


class TestCLIFunctionality:
    """Test CLI commands"""
    
    def test_status_command(self):
        """Test status command"""
        runner = CliRunner()
        result = runner.invoke(main, [
            'train', 'status', '--service', 'frontend', '--config-file', 'config/resource-optimized.yaml'
        ])
        
        assert result.exit_code == 0, f"Status command failed: {result.output}"
        assert "Total Experiments" in result.output, "Should show experiment count"
    
    def test_parallel_experiments_command_help(self):
        """Test parallel experiments command help"""
        runner = CliRunner()
        result = runner.invoke(main, [
            'train', 'parallel-experiments', '--help'
        ])
        
        assert result.exit_code == 0, "Help command should work"
        assert "--services" in result.output, "Should show services option"
        assert "--max-workers" in result.output, "Should show max-workers option"
    
    def test_clean_experiments_command(self):
        """Test clean experiments command"""
        runner = CliRunner()
        result = runner.invoke(main, [
            'train', 'clean-experiments', '--service', 'frontend', '--config-file', 'config/resource-optimized.yaml'
        ])
        
        assert result.exit_code == 0, f"Clean experiments command failed: {result.output}"


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow"""
    
    def test_short_experiment_workflow(self):
        """Test a short end-to-end experiment"""
        with open('config/resource-optimized.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create a minimal config for testing
        test_config = config.copy()
        test_config['experiment_duration_minutes'] = 1  # Very short test
        test_config['replica_counts'] = [1]  # Single replica
        test_config['load_levels'] = [5]  # Low load
        test_config['test_scenarios'] = ['browsing']  # Single scenario
        
        pipeline = DataAcquisitionPipeline(test_config)
        
        # Test that we can create experiment queue
        services = ['frontend']
        # This would normally create experiments, but we'll just test the setup
        assert hasattr(pipeline, 'run_parallel_training_experiments'), \
            "Pipeline should support parallel execution"
    
    def test_data_persistence_structure(self):
        """Test that data persistence structure is correct"""
        # Ensure training_data directory exists
        os.makedirs('training_data', exist_ok=True)
        
        # Test that we can create the expected file structure
        test_data = {
            'experiment_id': 'test_experiment',
            'service': 'frontend',
            'scenario': 'browsing',
            'replicas': 1,
            'users': 10,
            'metrics': {
                'cpu_cores': 'test_data.csv',
                'mem_bytes': 'test_data.csv'
            }
        }
        
        # Test JSON serialization
        json_str = json.dumps(test_data, indent=2, default=str)
        assert 'test_experiment' in json_str, "JSON serialization should work"
        
        # Test file creation
        test_file = 'training_data/test_experiment.json'
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2, default=str)
        
        assert os.path.exists(test_file), "Should be able to create data files"
        
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)


class TestSystemIntegration:
    """Test system integration and error handling"""
    
    def test_prometheus_query_error_handling(self):
        """Test that Prometheus query errors are handled gracefully"""
        client = PrometheusClient('http://localhost:9090')
        
        # Test with invalid query
        try:
            result = client.custom_query('invalid_query_that_should_fail')
            # Should not crash, even if it returns empty result
            assert result is not None or result == [], "Should handle invalid queries gracefully"
        except Exception as e:
            # If it raises an exception, it should be a known type
            assert isinstance(e, (ValueError, ConnectionError, TimeoutError)), \
                f"Unexpected exception type: {type(e)}"
    
    def test_concurrent_access_safety(self):
        """Test that concurrent access to shared resources is safe"""
        import threading
        import time
        
        results = []
        errors = []
        
        def test_worker(worker_id):
            try:
                client = PrometheusClient('http://localhost:9090')
                result = client.test_connection()
                results.append((worker_id, result))
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=test_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=10)
        
        # Verify results
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        
        # All should have successful connections
        for worker_id, result in results:
            assert result is True, f"Worker {worker_id} connection failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

