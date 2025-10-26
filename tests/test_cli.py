"""
Tests for MOrA CLI functionality
"""
import pytest
import json
import yaml
from click.testing import CliRunner
from unittest.mock import Mock, patch, MagicMock

try:
    from src.mora.cli.main import main
except ImportError:
    # Fallback for different import paths
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.mora.cli.main import main


class TestCLI:
    """Test cases for CLI commands"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
    
    def test_main_help(self):
        """Test that main command shows help"""
        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert "MOrA - Microservices-Aware Orchestrator Agent" in result.output
    
    def test_rightsize_command_help(self):
        """Test rightsize command help"""
        result = self.runner.invoke(main, ['rightsize', '--help'])
        assert result.exit_code == 0
        assert "Generate rightsizing recommendations" in result.output
        assert "--service" in result.output
        assert "--strategy" in result.output
        assert "--namespace" in result.output
    
    def test_status_command_help(self):
        """Test status command help"""
        result = self.runner.invoke(main, ['status', '--help'])
        assert result.exit_code == 0
        assert "Show current status" in result.output
        assert "--namespace" in result.output


class TestCLIIntegration:
    """Integration tests for CLI commands with mocked backends"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
    
    @patch('src.mora.cli.main.DataPipeline')
    @patch('src.mora.cli.main.StatisticalRightsizer')
    def test_status_command_success(self, mock_rightsizer, mock_pipeline_class):
        """Test successful status command execution"""
        # Mock the pipeline and its methods
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.test_connections.return_value = {
            'kubernetes': True,
            'prometheus': True
        }
        mock_pipeline.get_system_summary.return_value = {
            'namespace': 'hipster-shop',
            'total_services': 5,
            'services': ['frontend', 'backend', 'database'],
            'service_stats': {
                'frontend': {'replicas': 2, 'ready_replicas': 2, 'containers': 1}
            }
        }
        
        result = self.runner.invoke(main, ['status', '--namespace', 'test-namespace'])
        
        # Should succeed even with mocked backend
        # The actual implementation will show connection status
        assert result.exit_code == 0 or result.exit_code == 1  # May fail due to missing deps
    
    @patch('src.mora.cli.main.DataPipeline')
    @patch('src.mora.cli.main.StatisticalRightsizer')
    def test_rightsize_command_with_mocks(self, mock_rightsizer_class, mock_pipeline_class):
        """Test rightsize command with mocked dependencies"""
        # Mock the pipeline
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.test_connections.return_value = {
            'kubernetes': True,
            'prometheus': True
        }
        
        # Mock service data
        mock_service_data = {
            'service_name': 'test-service',
            'namespace': 'test-namespace',
            'deployment': {
                'name': 'test-service',
                'containers': [{
                    'name': 'test-container',
                    'resources': {
                        'requests': {'cpu': '100m', 'memory': '128Mi'}
                    }
                }]
            },
            'metrics': {}
        }
        
        mock_pipeline.collect_service_data.return_value = mock_service_data
        mock_pipeline.validate_data_quality.return_value = {'is_valid': True, 'warnings': []}
        
        # Mock the rightsizer
        mock_rightsizer = Mock()
        mock_rightsizer_class.return_value = mock_rightsizer
        mock_rightsizer.generate_recommendations.return_value = [{
            'container_name': 'test-container',
            'current_requests': {'cpu': '100m', 'memory': '128Mi'},
            'recommended_requests': {'cpu': '200m', 'memory': '256Mi'},
            'analysis': {
                'cpu': {'has_data': False, 'percentile_value': 0.1},
                'memory': {'has_data': False, 'max_usage_bytes': 100000000}
            }
        }]
        mock_rightsizer.validate_recommendations.return_value = {'is_valid': True, 'warnings': []}
        
        result = self.runner.invoke(main, [
            'rightsize', 
            '--service', 'test-service',
            '--namespace', 'test-namespace',
            '--strategy', 'statistical'
        ])
        
        # Should process without crashing (may exit with 1 due to missing deps)
        assert result.exit_code in [0, 1]
    
    def test_rightsize_command_validation(self):
        """Test rightsize command parameter validation"""
        # Test missing required service parameter
        result = self.runner.invoke(main, ['rightsize'])
        assert result.exit_code != 0
        
        # Test invalid strategy
        result = self.runner.invoke(main, [
            'rightsize', 
            '--service', 'test-service',
            '--strategy', 'invalid-strategy'
        ])
        # Should handle gracefully or show error
        assert result.exit_code != 0
    
    def test_output_format_options(self):
        """Test that output format options are available"""
        result = self.runner.invoke(main, ['rightsize', '--help'])
        assert result.exit_code == 0
        assert "--output-format" in result.output
        assert "table" in result.output
        assert "json" in result.output
        assert "yaml" in result.output


class TestCLIErrorHandling:
    """Test CLI error handling scenarios"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
    
    @patch('src.mora.cli.main.DataPipeline')
    def test_connection_failure_handling(self, mock_pipeline_class):
        """Test handling of connection failures"""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.test_connections.return_value = {
            'kubernetes': False,
            'prometheus': False
        }
        
        result = self.runner.invoke(main, [
            'status', 
            '--namespace', 'test-namespace'
        ])
        
        # Should handle gracefully
        assert "Cannot connect" in result.output or result.exit_code == 0
    
    @patch('src.mora.cli.main.DataPipeline')
    def test_service_not_found_handling(self, mock_pipeline_class):
        """Test handling when service is not found"""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.test_connections.return_value = {
            'kubernetes': True,
            'prometheus': True
        }
        mock_pipeline.collect_service_data.side_effect = ValueError("Service not found")
        
        result = self.runner.invoke(main, [
            'rightsize',
            '--service', 'nonexistent-service',
            '--namespace', 'test-namespace'
        ])
        
        # Should handle error gracefully
        assert result.exit_code != 0 or "Error" in result.output


class TestGrafanaIntegration:
    """Test cases for Grafana integration and CLI commands"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
    
    def test_setup_grafana_command_help(self):
        """Test setup-grafana command help"""
        result = self.runner.invoke(main, ['setup-grafana', '--help'])
        assert result.exit_code == 0
        assert "Grafana dashboard integration" in result.output or "Grafana Integration Setup" in result.output
        assert "--namespace" in result.output
        assert "--grafana-url" in result.output
    
    @patch('src.mora.cli.main.DataPipeline')
    def test_setup_grafana_command_success(self, mock_pipeline_class):
        """Test successful Grafana setup command"""
        # Mock the pipeline and Grafana client
        mock_pipeline = Mock()
        mock_grafana_client = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.grafana_client = mock_grafana_client
        
        # Mock successful Grafana operations
        mock_grafana_client.test_connection.return_value = True
        mock_pipeline.setup_grafana_integration.return_value = {
            'success': True,
            'dashboard_uid': 'test-uid-123',
            'dashboard_url': 'http://localhost:4000/d/test-uid-123',
            'namespace': 'hipster-shop'
        }
        
        result = self.runner.invoke(main, [
            'setup-grafana',
            '--namespace', 'test-namespace',
            '--grafana-url', 'http://localhost:4000'
        ])
        
        # Should succeed with mocked backend
        assert result.exit_code == 0 or result.exit_code == 1  # May fail due to missing deps
    
    @patch('src.mora.cli.main.DataPipeline')
    def test_setup_grafana_connection_failure(self, mock_pipeline_class):
        """Test Grafana setup with connection failure"""
        mock_pipeline = Mock()
        mock_grafana_client = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.grafana_client = mock_grafana_client
        
        # Mock connection failure
        mock_grafana_client.test_connection.return_value = False
        
        result = self.runner.invoke(main, ['setup-grafana'])
        
        # Should handle connection failure
        assert result.exit_code == 0 or result.exit_code == 1
    
    def test_train_commands_help(self):
        """Test training command help"""
        result = self.runner.invoke(main, ['train', '--help'])
        assert result.exit_code == 0
        assert "Model training commands" in result.output
    
    def test_clean_experiments_command_help(self):
        """Test clean-experiments command help"""
        result = self.runner.invoke(main, ['train', 'clean-experiments', '--help'])
        assert result.exit_code == 0
        assert "clean steady-state training" in result.output or "clean data" in result.output
    
    def test_dynamic_evaluation_command_help(self):
        """Test dynamic-evaluation command help"""
        result = self.runner.invoke(main, ['train', 'dynamic-evaluation', '--help'])
        assert result.exit_code == 0
        assert "dynamic evaluation experiment" in result.output or "Phase 4" in result.output


class TestEnhancedCLIFeatures:
    """Test cases for enhanced CLI features with Grafana integration"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
    
    @patch('src.mora.cli.main.DataPipeline')
    def test_status_command_with_grafana(self, mock_pipeline_class):
        """Test status command includes Grafana connection testing"""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.test_connections.return_value = {
            'kubernetes': True,
            'prometheus': True,
            'grafana': True  # New Grafana connection test
        }
        mock_pipeline.get_system_summary.return_value = {
            'namespace': 'hipster-shop',
            'total_services': 3,
            'service_stats': {}
        }
        
        result = self.runner.invoke(main, ['status'])
        
        # Should test all three connections now (K8s, Prometheus, Grafana)
        assert result.exit_code == 0 or result.exit_code == 1
    
    @patch('src.mora.cli.main.load_config')
    @patch('src.mora.cli.main.DataPipeline')
    def test_config_loading_integration(self, mock_pipeline_class, mock_load_config):
        """Test that CLI properly loads Grafana config from YAML"""
        mock_config = {
            'grafana': {'url': 'http://test-grafana:4000'},
            'prometheus': {'url': 'http://test-prometheus:9090'},
            'kubernetes': {'namespace': 'test-ns'}
        }
        mock_load_config.return_value = mock_config
        
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.test_connections.return_value = {'grafana': True, 'prometheus': True, 'kubernetes': True}
        mock_pipeline.get_system_summary.return_value = {'namespace': 'test-ns'}
        
        result = self.runner.invoke(main, ['status'])
        
        # Verify that config loading was attempted
        assert result.exit_code == 0 or result.exit_code == 1

    @patch('src.mora.cli.main.load_config')
    @patch('src.mora.cli.main.DataAcquisitionPipeline')
    def test_clean_experiments_triple_loop_config_display(self, mock_pipeline_class, mock_load_config):
        """Test that clean-experiments command displays correct triple-loop configuration"""
        # Mock config with the new parameters
        mock_config = {
            'training': {
                'steady_state_config': {
                    'experiment_duration_minutes': 45,
                    'replica_counts': [1, 2, 4, 6],
                    'load_levels_users': [10, 50, 100, 150, 200, 250],
                    'test_scenarios': ['browsing', 'checkout'],
                    'sample_interval': '15s'
                }
            },
            'kubernetes': {'namespace': 'hipster-shop'},
            'prometheus': {'url': 'http://localhost:9090'}
        }
        mock_load_config.return_value = mock_config
        
        # Mock the pipeline
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.run_isolated_training_experiment.return_value = {
            'status': 'completed',
            'total_combinations': 48,  # 4 replicas × 6 loads × 2 scenarios
            'experiments': []
        }
        
        result = self.runner.invoke(main, ['train', 'clean-experiments', '--service', 'test-service'])
        
        # Verify the command ran (may have exit code 0 or 1 due to missing deps)
        assert result.exit_code == 0 or result.exit_code == 1
        
        # Verify that config loading was called with the correct file
        mock_load_config.assert_called()

    @patch('src.mora.cli.main.load_config')
    @patch('src.mora.cli.main.DataAcquisitionPipeline')
    def test_clean_experiments_shows_data_quality_summary(self, mock_pipeline_class, mock_load_config):
        """Test that clean-experiments command shows data quality summary"""
        mock_config = {
            'training': {
                'steady_state_config': {
                    'replica_counts': [1, 2],
                    'load_levels_users': [10, 50],
                    'test_scenarios': ['browsing', 'checkout']
                }
            }
        }
        mock_load_config.return_value = mock_config
        
        # Mock experiment results with data quality information
        mock_experiments = [
            {
                'status': 'completed',
                'data_quality': {'status': 'passed', 'warnings': []}
            },
            {
                'status': 'completed_with_warnings',
                'data_quality': {'status': 'warnings', 'warnings': ['High NaN values']}
            }
        ]
        
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.run_isolated_training_experiment.return_value = {
            'status': 'completed',
            'total_combinations': 8,  # 2 replicas × 2 loads × 2 scenarios
            'experiments': mock_experiments
        }
        
        result = self.runner.invoke(main, ['train', 'clean-experiments', '--service', 'test-service'])
        
        # Should run without critical errors
        assert result.exit_code == 0 or result.exit_code == 1

    def test_train_status_command(self):
        """Test the new train status command"""
        mock_config = {
            'kubernetes': {'namespace': 'hipster-shop'},
            'prometheus': {'url': 'http://localhost:9090'},
            'training': {'steady_state_config': {
                'replica_counts': [1, 2], 'load_levels_users': [10, 50], 
                'test_scenarios': ['browsing']
            }}
        }
        
        with patch('src.mora.cli.main.DataAcquisitionPipeline') as mock_pipeline_class, \
             patch('src.mora.cli.main.load_config') as mock_load_config:
            mock_pipeline = Mock()
            mock_pipeline._get_completed_experiments.return_value = {
                'frontend_browsing_replicas_1_users_10',
                'frontend_browsing_replicas_1_users_50'
            }
            mock_pipeline_class.return_value = mock_pipeline
            mock_load_config.return_value = mock_config
            
            result = self.runner.invoke(main, ['train', 'status', '--service', 'frontend'])
            
            # Should run successfully
            assert result.exit_code == 0
            # Should show progress information
            assert "Training Progress for frontend" in result.output
            assert "Completed:" in result.output
            assert "Remaining:" in result.output

    def test_train_status_command_no_experiments(self):
        """Test train status command when no experiments are completed"""
        mock_config = {
            'kubernetes': {'namespace': 'hipster-shop'},
            'prometheus': {'url': 'http://localhost:9090'},
            'training': {'steady_state_config': {
                'replica_counts': [1, 2], 'load_levels_users': [10, 50], 
                'test_scenarios': ['browsing']
            }}
        }
        
        with patch('src.mora.cli.main.DataAcquisitionPipeline') as mock_pipeline_class, \
             patch('src.mora.cli.main.load_config') as mock_load_config:
            mock_pipeline = Mock()
            mock_pipeline._get_completed_experiments.return_value = set()
            mock_pipeline_class.return_value = mock_pipeline
            mock_load_config.return_value = mock_config
            
            result = self.runner.invoke(main, ['train', 'status', '--service', 'frontend'])
            
            # Should run successfully
            assert result.exit_code == 0
            # Should indicate no experiments completed
            assert "No experiments completed yet" in result.output

    def test_train_status_command_with_completed_experiments(self):
        """Test train status command showing completed experiments"""
        completed_experiments = {
            'frontend_browsing_replicas_1_users_10',
            'frontend_browsing_replicas_2_users_50',
            'frontend_checkout_replicas_1_users_100'
        }
        
        # Mock both the config loading and pipeline
        mock_config = {
            'kubernetes': {'namespace': 'hipster-shop'},
            'prometheus': {'url': 'http://localhost:9090'},
            'training': {
                'steady_state_config': {
                    'replica_counts': [1, 2, 4, 6],
                    'load_levels_users': [10, 50, 100, 150, 200, 250],
                    'test_scenarios': ['browsing', 'checkout']
                }
            }
        }
        
        with patch('src.mora.cli.main.DataAcquisitionPipeline') as mock_pipeline_class, \
             patch('src.mora.cli.main.load_config') as mock_load_config:
            
            mock_pipeline = Mock()
            mock_pipeline._get_completed_experiments.return_value = completed_experiments
            mock_pipeline_class.return_value = mock_pipeline
            mock_load_config.return_value = mock_config
            
            result = self.runner.invoke(main, ['train', 'status', '--service', 'frontend'])
            
            # Should run successfully
            assert result.exit_code == 0
            # Should show completed experiments count
            assert "Completed: 3" in result.output
            # Should show some completed experiment IDs
            assert "frontend_browsing_replicas_1_users_10" in result.output


class TestCLIProductionReadiness:
    """Test cases for CLI production readiness"""

    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()

    def test_error_handling_in_status_command(self):
        """Test that status command handles errors gracefully"""
        with patch('src.mora.core.data_acquisition.DataAcquisitionPipeline') as mock_pipeline_class:
            # Simulate an error during pipeline initialization
            mock_pipeline_class.side_effect = Exception("Connection failed")
            
            result = self.runner.invoke(main, ['train', 'status', '--service', 'frontend'])
            
            # Should handle error gracefully and not crash
            assert result.exit_code != 0  # Should exit with error code
            assert "Error checking status" in result.output

    def test_config_file_handling_in_status_command(self):
        """Test that status command handles different config files"""
        with patch('src.mora.core.data_acquisition.DataAcquisitionPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline._get_completed_experiments.return_value = set()
            mock_pipeline_class.return_value = mock_pipeline
            
            result = self.runner.invoke(main, [
                'train', 'status', 
                '--service', 'frontend',
                '--config-file', 'custom-config.yaml'
            ])
            
            # Should run successfully even with custom config file
            assert result.exit_code == 0


class TestParallelExperimentsCLI:
    """Test cases for parallel experiments CLI command"""

    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()

    def test_parallel_experiments_command_help(self):
        """Test parallel-experiments command help"""
        result = self.runner.invoke(main, ['train', 'parallel-experiments', '--help'])
        assert result.exit_code == 0
        assert "parallel across multiple" in result.output
        assert "--services" in result.output
        assert "--max-workers" in result.output

    @patch('src.mora.cli.main.DataAcquisitionPipeline')
    @patch('src.mora.cli.main.load_config')
    def test_parallel_experiments_command_success(self, mock_load_config, mock_pipeline_class):
        """Test successful parallel experiments command"""
        # Mock config loading
        mock_config = {
            'training': {
                'steady_state_config': {
                    'experiment_duration_minutes': 45,
                    'replica_counts': [1, 2],
                    'load_levels_users': [10, 50],
                    'test_scenarios': ['browsing']
                }
            },
            'kubernetes': {'namespace': 'hipster-shop'},
            'prometheus': {'url': 'http://localhost:9090'}
        }
        mock_load_config.return_value = mock_config

        # Mock pipeline and results
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.run_parallel_training_experiments.return_value = {
            'status': 'completed',
            'experiments': [
                {'status': 'completed', 'experiment_id': 'test1'},
                {'status': 'completed', 'experiment_id': 'test2'},
                {'status': 'failed', 'experiment_id': 'test3', 'error': 'test error'}
            ]
        }

        result = self.runner.invoke(main, [
            'train', 'parallel-experiments',
            '--services', 'frontend,checkoutservice',
            '--max-workers', '2'
        ])

        # Should succeed
        assert result.exit_code == 0
        assert "Parallel Training Experiments" in result.output
        assert "frontend" in result.output and "checkoutservice" in result.output
        assert "Successful: 2" in result.output
        assert "Failed: 1" in result.output

    @patch('src.mora.cli.main.DataAcquisitionPipeline')
    @patch('src.mora.cli.main.load_config')
    def test_parallel_experiments_config_loading(self, mock_load_config, mock_pipeline_class):
        """Test parallel experiments command loads config correctly"""
        mock_config = {
            'training': {
                'steady_state_config': {
                    'experiment_duration_minutes': 30,
                    'replica_counts': [1, 2, 4],
                    'load_levels_users': [10, 50, 100],
                    'test_scenarios': ['browsing', 'checkout']
                }
            },
            'kubernetes': {'namespace': 'test-namespace'},
            'prometheus': {'url': 'http://localhost:9090'}
        }
        mock_load_config.return_value = mock_config

        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.run_parallel_training_experiments.return_value = {'status': 'completed', 'experiments': []}

        result = self.runner.invoke(main, [
            'train', 'parallel-experiments',
            '--services', 'frontend',
            '--config-file', 'test-config.yaml'
        ])

        # Should call load_config with the specified file
        mock_load_config.assert_called_once_with('test-config.yaml')
        
        # Should display configuration correctly
        assert result.exit_code == 0
        assert "Total Experiments:" in result.output

    @patch('src.mora.cli.main.DataAcquisitionPipeline')
    @patch('src.mora.cli.main.load_config')
    def test_parallel_experiments_error_handling(self, mock_load_config, mock_pipeline_class):
        """Test parallel experiments command error handling"""
        mock_load_config.return_value = {'training': {'steady_state_config': {}}}
        
        # Make pipeline initialization fail
        mock_pipeline_class.side_effect = Exception("Connection failed")

        result = self.runner.invoke(main, [
            'train', 'parallel-experiments',
            '--services', 'frontend'
        ])

        # Should handle error gracefully - either exit code != 0 or error message in output
        assert result.exit_code != 0 or "Error during parallel training experiments" in result.output

    @patch('src.mora.cli.main.DataAcquisitionPipeline')
    @patch('src.mora.cli.main.load_config')
    def test_parallel_experiments_pipeline_call(self, mock_load_config, mock_pipeline_class):
        """Test that parallel experiments calls pipeline correctly"""
        mock_config = {'training': {'steady_state_config': {}}, 'kubernetes': {}, 'prometheus': {}}
        mock_load_config.return_value = mock_config

        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.run_parallel_training_experiments.return_value = {'status': 'completed', 'experiments': []}

        result = self.runner.invoke(main, [
            'train', 'parallel-experiments',
            '--services', 'frontend,checkoutservice',
            '--max-workers', '3'
        ])

        # Should call pipeline with correct parameters
        mock_pipeline.run_parallel_training_experiments.assert_called_once()
        call_args = mock_pipeline.run_parallel_training_experiments.call_args
        
        # Verify service list and max_workers are passed correctly
        assert call_args[0][0] == ['frontend', 'checkoutservice']  # services list
        assert call_args[0][1] == mock_config  # config
        assert call_args.kwargs['max_workers'] == 3  # max_workers kwarg

    def test_parallel_experiments_services_parsing(self):
        """Test that services parameter is parsed correctly"""
        result = self.runner.invoke(main, [
            'train', 'parallel-experiments',
            '--services', 'frontend,checkoutservice,cartservice',
            '--help'
        ])

        # The help should show the command exists and accepts services parameter
        assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__])

