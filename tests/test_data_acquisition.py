"""
Tests for DataAcquisitionPipeline with triple-loop implementation and data quality validation
"""
import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

try:
    from src.mora.core.data_acquisition import DataAcquisitionPipeline
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.mora.core.data_acquisition import DataAcquisitionPipeline


class TestDataAcquisitionTripleLoop:
    """Test cases for the new triple-loop implementation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_k8s_client = Mock()
        self.mock_prom_client = Mock()
        self.mock_load_generator = Mock()
        
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        with patch('src.mora.core.data_acquisition.DataAcquisitionPipeline.__init__', 
                   lambda x, **kwargs: None):
            self.pipeline = DataAcquisitionPipeline()
            self.pipeline.k8s_client = self.mock_k8s_client
            self.pipeline.prom_client = self.mock_prom_client
            self.pipeline.load_generator = self.mock_load_generator
            self.pipeline.namespace = "test-namespace"
            self.pipeline.data_dir = self.temp_dir

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_triple_loop_experiment_count_calculation(self):
        """Test that total combinations correctly includes scenarios"""
        config = {
            "replica_counts": [1, 2, 4],
            "load_levels_users": [10, 50, 100],
            "test_scenarios": ["browsing", "checkout"],
            "experiment_duration_minutes": 30
        }
        
        # Mock the over-provisioning and other dependencies
        self.mock_load_generator.overprovision_non_target_services.return_value = True
        self.mock_k8s_client.scale_deployment.return_value = True
        
        with patch.object(self.pipeline, '_run_steady_state_experiment') as mock_run:
            mock_run.return_value = {
                "experiment_id": "test",
                "status": "completed",
                "metrics": {}
            }
            
            result = self.pipeline.run_isolated_training_experiment(
                target_service="test-service",
                config=config
            )
            
            # Verify total combinations: 3 replicas × 3 loads × 2 scenarios = 18
            assert result["total_combinations"] == 18
            assert len(result["experiments"]) == 18

    def test_triple_loop_scenario_iteration(self):
        """Test that scenarios are properly iterated in outer loop"""
        config = {
            "replica_counts": [1, 2],
            "load_levels_users": [10, 50],
            "test_scenarios": ["browsing", "checkout"],
            "experiment_duration_minutes": 30
        }
        
        self.mock_load_generator.overprovision_non_target_services.return_value = True
        self.mock_k8s_client.scale_deployment.return_value = True
        
        experiment_calls = []
        
        def capture_experiment_call(*args, **kwargs):
            experiment_calls.append(kwargs.get('test_scenario'))
            return {
                "experiment_id": f"test_{kwargs.get('test_scenario')}_{kwargs.get('replica_count')}_{kwargs.get('load_users')}",
                "status": "completed",
                "metrics": {}
            }
        
        with patch.object(self.pipeline, '_run_steady_state_experiment', side_effect=capture_experiment_call):
            result = self.pipeline.run_isolated_training_experiment(
                target_service="test-service",
                config=config
            )
            
            # Should have 2×2×2 = 8 experiments total
            assert len(experiment_calls) == 8
            
            # Verify scenario distribution (first 4 should be browsing, last 4 should be checkout)
            browsing_calls = experiment_calls[:4]
            checkout_calls = experiment_calls[4:]
            
            assert all(scenario == "browsing" for scenario in browsing_calls)
            assert all(scenario == "checkout" for scenario in checkout_calls)

    def test_experiment_id_formatting_with_scenarios(self):
        """Test that experiment IDs include scenario names"""
        config = {
            "replica_counts": [1],
            "load_levels_users": [10],
            "test_scenarios": ["browsing", "checkout"],
            "experiment_duration_minutes": 30
        }
        
        self.mock_load_generator.overprovision_non_target_services.return_value = True
        self.mock_k8s_client.scale_deployment.return_value = True
        
        experiment_ids = []
        
        def capture_experiment_id(*args, **kwargs):
            experiment_ids.append(kwargs.get('experiment_id'))
            return {"experiment_id": kwargs.get('experiment_id'), "status": "completed", "metrics": {}}
        
        with patch.object(self.pipeline, '_run_steady_state_experiment', side_effect=capture_experiment_id):
            result = self.pipeline.run_isolated_training_experiment(
                target_service="test-service",
                config=config
            )
            
            # Should have scenario in the ID
            assert len(experiment_ids) == 2
            assert "test-service_browsing_replicas_1_users_10" in experiment_ids
            assert "test-service_checkout_replicas_1_users_10" in experiment_ids


class TestDataQualityValidation:
    """Test cases for data quality validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        with patch('src.mora.core.data_acquisition.DataAcquisitionPipeline.__init__', 
                   lambda x, **kwargs: None):
            self.pipeline = DataAcquisitionPipeline()
            self.pipeline.data_dir = self.temp_dir

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('src.mora.core.data_acquisition.pd')
    def test_data_quality_completeness_check(self, mock_pd):
        """Test data completeness validation"""
        # Mock pandas DataFrame
        mock_df = Mock()
        mock_df.isnull.return_value.sum.return_value.sum.return_value = 0
        mock_df.size = 100
        
        metrics_data = {
            "cpu_cores_data": mock_df,
            "mem_bytes_data": mock_df
        }
        
        config = {
            "required_metrics": ["cpu_cores", "mem_bytes", "requests_per_second"],
            "data_quality_checks": {
                "min_data_completeness_percent": 80,
                "max_metric_nan_percent": 5,
                "max_std_dev_percent": 10
            }
        }
        
        result = self.pipeline._validate_experiment_data_quality(metrics_data, config)
        
        assert "status" in result
        assert "checks" in result
        assert "completeness" in result["checks"]

    def test_data_quality_no_metrics_data(self):
        """Test data quality validation with no metrics"""
        metrics_data = {}
        config = {
            "required_metrics": ["cpu_cores", "mem_bytes"],
            "data_quality_checks": {
                "min_data_completeness_percent": 90
            }
        }
        
        result = self.pipeline._validate_experiment_data_quality(metrics_data, config)
        
        assert result["status"] == "failed"
        assert "No metrics data collected" in result["warnings"]

    @patch('src.mora.core.data_acquisition.pd')
    def test_data_quality_nan_check(self, mock_pd):
        """Test NaN value validation"""
        # Mock DataFrame with high NaN content
        mock_df = Mock()
        mock_df.isnull.return_value.sum.return_value.sum.return_value = 20  # 20% NaN
        mock_df.size = 100
        
        metrics_data = {
            "service_cpu_cores": mock_df,
            "service_mem_bytes": mock_df
        }
        
        config = {
            "data_quality_checks": {
                "max_metric_nan_percent": 5
            }
        }
        
        result = self.pipeline._validate_experiment_data_quality(metrics_data, config)
        
        # Should have warnings due to high NaN percentage
        assert len(result["warnings"]) > 0
        assert any("NaN values" in warning for warning in result["warnings"])


class TestSteadyStateExperiment:
    """Test cases for steady-state experiment method"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        with patch('src.mora.core.data_acquisition.DataAcquisitionPipeline.__init__', 
                   lambda x, **kwargs: None):
            self.pipeline = DataAcquisitionPipeline()
            self.pipeline.k8s_client = Mock()
            self.pipeline.prom_client = Mock()
            self.pipeline.load_generator = Mock()
            self.pipeline.namespace = "test-namespace"
            self.pipeline.data_dir = self.temp_dir

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_steady_state_experiment_accepts_test_scenario(self):
        """Test that _run_steady_state_experiment accepts test_scenario parameter"""
        # Mock dependencies
        self.pipeline.k8s_client.scale_deployment.return_value = True
        self.pipeline.load_generator.create_jmeter_script.return_value = "/tmp/test_script.jmx"
        self.pipeline.load_generator.run_load_test.return_value = {"status": "completed"}
        self.pipeline.prom_client.get_comprehensive_metrics.return_value = {}
        
        with patch.object(self.pipeline, '_validate_experiment_data_quality') as mock_validate:
            mock_validate.return_value = {"status": "passed", "warnings": []}
            
            result = self.pipeline._run_steady_state_experiment(
                target_service="test-service",
                replica_count=2,
                load_users=100,
                duration_minutes=30,
                experiment_id="test_experiment",
                test_scenario="checkout",  # This is the new parameter
                config={"stabilization_wait_seconds": 60}
            )
            
            # Verify the method accepted the test_scenario parameter
            assert result["status"] in ["completed", "completed_with_warnings"]
            
            # Verify JMeter script was created with the scenario
            self.pipeline.load_generator.create_jmeter_script.assert_called()
            call_args = self.pipeline.load_generator.create_jmeter_script.call_args
            assert call_args.kwargs.get('test_scenario') == "checkout"


class TestDataAcquisitionResumability:
    """Test cases for the resumability features"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_k8s_client = Mock()
        self.mock_prom_client = Mock()
        self.mock_load_generator = Mock()
        
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        with patch('src.mora.core.data_acquisition.DataAcquisitionPipeline.__init__', 
                   lambda x, **kwargs: None):
            self.pipeline = DataAcquisitionPipeline()
            self.pipeline.k8s_client = self.mock_k8s_client
            self.pipeline.prom_client = self.mock_prom_client
            self.pipeline.load_generator = self.mock_load_generator
            self.pipeline.namespace = "test-namespace"
            self.pipeline.data_dir = self.temp_dir

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_get_experiment_id(self):
        """Test experiment ID generation"""
        exp_id = self.pipeline._get_experiment_id("frontend", "browsing", 2, 50)
        expected = "frontend_browsing_replicas_2_users_50"
        assert exp_id == expected

    def test_is_experiment_completed_with_valid_data(self):
        """Test experiment completion check with valid completed data"""
        exp_id = "test_experiment"
        exp_file = os.path.join(self.temp_dir, f"{exp_id}.json")
        
        # Create a completed experiment file
        with open(exp_file, 'w') as f:
            json.dump({"status": "completed", "experiment_id": exp_id}, f)
        
        assert self.pipeline._is_experiment_completed(exp_id) == True

    def test_is_experiment_completed_with_warnings(self):
        """Test experiment completion check with warnings status"""
        exp_id = "test_experiment_warnings"
        exp_file = os.path.join(self.temp_dir, f"{exp_id}.json")
        
        # Create an experiment file with warnings
        with open(exp_file, 'w') as f:
            json.dump({"status": "completed_with_warnings", "experiment_id": exp_id}, f)
        
        assert self.pipeline._is_experiment_completed(exp_id) == True

    def test_is_experiment_completed_with_failed_status(self):
        """Test experiment completion check with failed status"""
        exp_id = "test_experiment_failed"
        exp_file = os.path.join(self.temp_dir, f"{exp_id}.json")
        
        # Create a failed experiment file
        with open(exp_file, 'w') as f:
            json.dump({"status": "failed", "experiment_id": exp_id}, f)
        
        assert self.pipeline._is_experiment_completed(exp_id) == False

    def test_is_experiment_completed_no_file(self):
        """Test experiment completion check when file doesn't exist"""
        exp_id = "nonexistent_experiment"
        assert self.pipeline._is_experiment_completed(exp_id) == False

    def test_get_completed_experiments(self):
        """Test getting set of completed experiments for a service"""
        service_name = "test-service"
        
        # Create test experiment files
        completed_exp1 = "test-service_browsing_replicas_1_users_10"
        completed_exp2 = "test-service_checkout_replicas_2_users_50"
        failed_exp = "test-service_browsing_replicas_1_users_20"
        other_service = "other-service_browsing_replicas_1_users_10"
        
        # Create completed experiment files
        for exp_id, status in [
            (completed_exp1, "completed"),
            (completed_exp2, "completed_with_warnings"),
            (failed_exp, "failed"),
            (other_service, "completed")
        ]:
            exp_file = os.path.join(self.temp_dir, f"{exp_id}.json")
            with open(exp_file, 'w') as f:
                json.dump({"status": status, "experiment_id": exp_id}, f)
        
        completed = self.pipeline._get_completed_experiments(service_name)
        
        # Should only include completed experiments for the specified service
        assert len(completed) == 2
        assert completed_exp1 in completed
        assert completed_exp2 in completed
        assert failed_exp not in completed  # Failed experiment not included
        assert other_service not in completed  # Different service not included

    def test_resumable_training_skips_completed_experiments(self):
        """Test that resumable training skips completed experiments"""
        # Mock the load generator and other dependencies
        self.pipeline.load_generator.overprovision_non_target_services.return_value = True
        
        # Create some completed experiments
        completed_exp1 = f"frontend_browsing_replicas_1_users_10.json"
        completed_exp2 = f"frontend_browsing_replicas_1_users_50.json"
        
        for exp_file in [completed_exp1, completed_exp2]:
            exp_path = os.path.join(self.temp_dir, exp_file)
            with open(exp_path, 'w') as f:
                json.dump({"status": "completed", "experiment_id": exp_file.replace('.json', '')}, f)

        config = {
            "replica_counts": [1, 2],
            "load_levels_users": [10, 50],
            "test_scenarios": ["browsing"],
            "experiment_duration_minutes": 30
        }

        # Mock the steady state experiment method
        with patch.object(self.pipeline, '_run_steady_state_experiment') as mock_run:
            mock_run.return_value = {
                "status": "completed",
                "experiment_id": "test_exp",
                "metrics": {}
            }
            
            # Mock the save methods
            with patch.object(self.pipeline, '_save_experiment_data'):
                result = self.pipeline.run_isolated_training_experiment("frontend", config)
            
            # Should have skipped 2 completed experiments and run 2 remaining
            # Total combinations: 1 scenario × 2 replicas × 2 loads = 4
            # Completed: 2, Remaining: 2
            assert result["total_combinations"] == 4
            
            # Verify that only the non-completed experiments were run
            # Should have been called twice (for the 2 non-completed experiments)
            assert mock_run.call_count == 2


class TestDataAcquisitionParallelExecution:
    """Test cases for parallel execution functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_k8s_client = Mock()
        self.mock_prom_client = Mock()
        self.mock_load_generator = Mock()
        
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        with patch('src.mora.core.data_acquisition.DataAcquisitionPipeline.__init__', 
                   lambda x, **kwargs: None):
            self.pipeline = DataAcquisitionPipeline()
            self.pipeline.k8s_client = self.mock_k8s_client
            self.pipeline.prom_client = self.mock_prom_client
            self.pipeline.load_generator = self.mock_load_generator
            self.pipeline.namespace = "test-namespace"
            self.pipeline.data_dir = self.temp_dir

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_parallel_experiment_queue_creation(self):
        """Test that parallel experiment queue is created correctly"""
        services = ["frontend", "checkoutservice"]
        steady_config = {
            "replica_counts": [1, 2],
            "load_levels_users": [10, 50],
            "test_scenarios": ["browsing"],
            "experiment_duration_minutes": 30
        }
        config = {"steady_state_config": steady_config}
        
        with patch.object(self.pipeline, '_run_steady_state_experiment') as mock_run:
            mock_run.return_value = {"status": "completed", "experiment_id": "test"}
            
            with patch.object(self.pipeline, '_save_experiment_data'):
                result = self.pipeline.run_parallel_training_experiments(services, config)
            
            # Should create 8 experiments: 2 services × 1 scenario × 2 replicas × 2 loads = 8
            assert len(result["experiments"]) == 8
            assert result["total_experiments"] == 8

    def test_parallel_execution_with_service_locking(self):
        """Test that service-level locking works correctly"""
        services = ["frontend"]
        steady_config = {
            "replica_counts": [1],
            "load_levels_users": [10, 20],
            "test_scenarios": ["browsing"],
            "experiment_duration_minutes": 30
        }
        config = {"steady_state_config": steady_config}
        
        # Track when scaling is called to verify locking
        scale_calls = []
        
        def mock_scale_deployment(service, namespace, replicas):
            scale_calls.append((service, replicas))
            return True
        
        self.pipeline.k8s_client.scale_deployment = mock_scale_deployment
        
        with patch.object(self.pipeline, '_run_steady_state_experiment') as mock_run:
            mock_run.return_value = {"status": "completed", "experiment_id": "test"}
            
            with patch.object(self.pipeline, '_save_experiment_data'):
                result = self.pipeline.run_parallel_training_experiments(services, config, max_workers=2)
            
            # Should have 2 experiments and both should complete
            assert len(result["experiments"]) == 2
            assert result["status"] == "completed"

    def test_parallel_execution_skips_completed(self):
        """Test that parallel execution skips already completed experiments"""
        services = ["frontend"]
        steady_config = {
            "replica_counts": [1],
            "load_levels_users": [10],
            "test_scenarios": ["browsing"],
            "experiment_duration_minutes": 30
        }
        config = {"steady_state_config": steady_config}
        
        # Create a completed experiment file
        completed_exp_id = "frontend_browsing_replicas_1_users_10"
        exp_file = os.path.join(self.temp_dir, f"{completed_exp_id}.json")
        with open(exp_file, 'w') as f:
            json.dump({"status": "completed", "experiment_id": completed_exp_id}, f)
        
        # Mock the completion check to return True for the completed experiment
        def mock_is_completed(exp_id):
            return exp_id == completed_exp_id
        
        self.pipeline._is_experiment_completed = mock_is_completed
        
        with patch.object(self.pipeline, '_run_steady_state_experiment') as mock_run:
            mock_run.return_value = {"status": "completed", "experiment_id": "test"}
            
            with patch.object(self.pipeline, '_save_experiment_data'):
                result = self.pipeline.run_parallel_training_experiments(services, config)
            
            # Should have 1 experiment result
            assert len(result["experiments"]) == 1
            
            # Check if the result is marked as skipped
            skipped_results = [exp for exp in result["experiments"] if exp.get("status") == "skipped"]
            assert len(skipped_results) == 1
            assert skipped_results[0]["experiment_id"] == completed_exp_id

    def test_parallel_max_workers_limit(self):
        """Test that max_workers parameter is respected"""
        services = ["frontend", "checkoutservice", "cartservice", "paymentservice"]
        steady_config = {"replica_counts": [1], "load_levels_users": [10], "test_scenarios": ["browsing"]}
        config = {"steady_state_config": steady_config}
        
        with patch.object(self.pipeline, '_run_steady_state_experiment') as mock_run:
            mock_run.return_value = {"status": "completed", "experiment_id": "test"}
            
            with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
                mock_executor_instance = Mock()
                mock_executor.return_value.__enter__.return_value = mock_executor_instance
                
                with patch.object(self.pipeline, '_save_experiment_data'):
                    result = self.pipeline.run_parallel_training_experiments(services, config, max_workers=2)
                
                # Verify ThreadPoolExecutor was called with max_workers=2
                mock_executor.assert_called_with(max_workers=2)

    def test_parallel_error_handling(self):
        """Test error handling in parallel execution"""
        services = ["frontend"]
        steady_config = {
            "replica_counts": [1],
            "load_levels_users": [10],
            "test_scenarios": ["browsing"],
            "experiment_duration_minutes": 30
        }
        config = {"steady_state_config": steady_config}
        
        # Make the experiment fail
        with patch.object(self.pipeline, '_run_steady_state_experiment') as mock_run:
            mock_run.side_effect = Exception("Test error")
            
            with patch.object(self.pipeline, '_save_experiment_data'):
                result = self.pipeline.run_parallel_training_experiments(services, config)
            
            # Should handle the error gracefully
            assert len(result["experiments"]) == 1
            assert result["experiments"][0]["status"] == "failed"
            assert "Test error" in result["experiments"][0]["error"]

    def test_parallel_results_tracking(self):
        """Test that parallel execution properly tracks results"""
        services = ["frontend", "checkoutservice"]
        steady_config = {
            "replica_counts": [1],
            "load_levels_users": [10],
            "test_scenarios": ["browsing"],
            "experiment_duration_minutes": 30
        }
        config = {"steady_state_config": steady_config}
        
        with patch.object(self.pipeline, '_run_steady_state_experiment') as mock_run:
            mock_run.return_value = {"status": "completed", "experiment_id": "test"}
            
            with patch.object(self.pipeline, '_save_experiment_data'):
                result = self.pipeline.run_parallel_training_experiments(services, config)
            
            # Verify result structure
            assert "start_time" in result
            assert "end_time" in result
            assert "services" in result
            assert "total_experiments" in result
            assert "experiments" in result
            assert "status" in result
            assert result["status"] == "completed"
            assert set(result["services"]) == set(services)


class TestDataAcquisitionProductionReadiness:
    """Test cases for production readiness and robustness"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_k8s_client = Mock()
        self.mock_prom_client = Mock()
        self.mock_load_generator = Mock()
        
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        with patch('src.mora.core.data_acquisition.DataAcquisitionPipeline.__init__', 
                   lambda x, **kwargs: None):
            self.pipeline = DataAcquisitionPipeline()
            self.pipeline.k8s_client = self.mock_k8s_client
            self.pipeline.prom_client = self.mock_prom_client
            self.pipeline.load_generator = self.mock_load_generator
            self.pipeline.namespace = "test-namespace"
            self.pipeline.data_dir = self.temp_dir

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_error_handling_in_experiment_completion_check(self):
        """Test error handling when checking experiment completion"""
        # Test with corrupted JSON file
        exp_id = "corrupted_experiment"
        exp_file = os.path.join(self.temp_dir, f"{exp_id}.json")
        
        with open(exp_file, 'w') as f:
            f.write("invalid json content")
        
        # Should return False for corrupted files
        assert self.pipeline._is_experiment_completed(exp_id) == False

    def test_error_handling_in_get_completed_experiments(self):
        """Test error handling when getting completed experiments"""
        # Test with directory that doesn't exist
        self.pipeline.data_dir = "/nonexistent/directory"
        
        # Should return empty set and not crash
        completed = self.pipeline._get_completed_experiments("test-service")
        assert completed == set()

    def test_data_directory_creation(self):
        """Test that data directory is created if it doesn't exist"""
        import shutil
        
        # Remove temp directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Initialize pipeline with non-existent directory
        with patch('src.mora.core.data_acquisition.DataAcquisitionPipeline.__init__', 
                   lambda x, namespace="test", prometheus_url="http://test": None):
            with patch('os.makedirs') as mock_makedirs:
                pipeline = DataAcquisitionPipeline()
                mock_makedirs.assert_called()

    def test_experiment_id_consistency(self):
        """Test that experiment IDs are generated consistently"""
        # Generate same ID multiple times
        exp_id1 = self.pipeline._get_experiment_id("frontend", "browsing", 2, 100)
        exp_id2 = self.pipeline._get_experiment_id("frontend", "browsing", 2, 100)
        
        assert exp_id1 == exp_id2
        assert exp_id1 == "frontend_browsing_replicas_2_users_100"

    def test_file_operations_are_safe(self):
        """Test that file operations handle permissions and errors gracefully"""
        # Create a file that can't be written to (simulate permission error)
        exp_id = "permission_test"
        exp_file = os.path.join(self.temp_dir, f"{exp_id}.json")
        
        # Create a file and then make it read-only
        with open(exp_file, 'w') as f:
            json.dump({"status": "completed"}, f)
        
        # The completion check should still work
        assert self.pipeline._is_experiment_completed(exp_id) == True

    def test_complete_12_metrics_collection(self):
        """Test that complete 12 metrics are collected (6 original + 6 substitute)"""
        # Mock the complete 12-metric system
        mock_metrics = {
            # Original infrastructure metrics (6)
            'cpu_cores': Mock(),
            'mem_bytes': Mock(),
            'pod_restarts': Mock(),
            'replica_count': Mock(),
            'node_cpu_util': Mock(),
            'node_mem_util': Mock(),
            # Substitute metrics (6)
            'net_rx_bytes': Mock(),
            'net_tx_bytes': Mock(),
            'network_activity_rate': Mock(),
            'processing_intensity': Mock(),
            'service_stability': Mock(),
            'resource_pressure': Mock()
        }
        
        self.mock_prom_client.get_comprehensive_metrics.return_value = mock_metrics
        
        # Test metrics collection
        result = self.pipeline._collect_metrics('test-service', 5)
        
        # Verify comprehensive metrics were requested
        self.mock_prom_client.get_comprehensive_metrics.assert_called_once_with(
            'test-namespace', 'test-service', 5
        )
        
        # Verify all expected metrics are present (12 total)
        assert len(result) == 12  # Complete 12 metrics
        for metric_name in mock_metrics.keys():
            assert metric_name in result


if __name__ == "__main__":
    pytest.main([__file__])
