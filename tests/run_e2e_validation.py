#!/usr/bin/env python3
"""
Comprehensive End-to-End System Validation

This script runs a complete validation of the MOrA system to ensure
all components are working correctly before starting data collection.
"""

import sys
import os
import time
import subprocess
import requests
import yaml
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mora.monitoring.prometheus_client import PrometheusClient
from mora.core.data_acquisition import DataAcquisitionPipeline
from mora.core.load_generator import LoadGenerator
from mora.cli.main import main
from click.testing import CliRunner


class SystemValidator:
    """Comprehensive system validation"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
    
    def log(self, message, level="INFO"):
        """Log validation results"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def validate_minikube(self):
        """Validate Minikube cluster"""
        self.log("Validating Minikube cluster...")
        try:
            result = subprocess.run(['kubectl', 'cluster-info'], 
                                 capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise Exception(f"kubectl cluster-info failed: {result.stderr}")
            
            # Check for minikube in cluster info or context
            if "minikube" not in result.stdout.lower() and "minikube" not in result.stderr.lower():
                # Try to get current context
                context_result = subprocess.run(['kubectl', 'config', 'current-context'], 
                                              capture_output=True, text=True, timeout=5)
                if context_result.returncode == 0 and "minikube" not in context_result.stdout.lower():
                    raise Exception("Not connected to Minikube cluster")
            
            self.results['minikube'] = True
            self.log("✅ Minikube cluster is running")
            return True
            
        except Exception as e:
            self.errors.append(f"Minikube validation failed: {e}")
            self.log(f"❌ Minikube validation failed: {e}", "ERROR")
            return False
    
    def validate_prometheus(self):
        """Validate Prometheus connectivity"""
        self.log("Validating Prometheus connectivity...")
        try:
            # Test basic connectivity
            response = requests.get('http://localhost:9090/-/ready', timeout=5)
            if response.status_code != 200:
                raise Exception(f"Prometheus not ready: HTTP {response.status_code}")
            
            # Test Prometheus client
            client = PrometheusClient('http://localhost:9090')
            if not client.test_connection():
                raise Exception("Prometheus client connection failed")
            
            self.results['prometheus'] = True
            self.log("✅ Prometheus is accessible and responding")
            return True
            
        except Exception as e:
            self.errors.append(f"Prometheus validation failed: {e}")
            self.log(f"❌ Prometheus validation failed: {e}", "ERROR")
            return False
    
    def validate_hipster_shop(self):
        """Validate Hipster Shop deployment"""
        self.log("Validating Hipster Shop deployment...")
        try:
            result = subprocess.run(['kubectl', 'get', 'pods', '-n', 'hipster-shop'], 
                                 capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise Exception(f"Cannot access hipster-shop namespace: {result.stderr}")
            
            # Check for key services
            services = ['frontend', 'checkoutservice', 'cartservice']
            missing_services = []
            for service in services:
                if service not in result.stdout:
                    missing_services.append(service)
            
            if missing_services:
                raise Exception(f"Missing services: {missing_services}")
            
            # Count running pods
            lines = [line for line in result.stdout.split('\n') if 'Running' in line]
            if len(lines) < 5:  # Should have multiple services running
                self.warnings.append("Fewer than expected pods running")
            
            self.results['hipster_shop'] = True
            self.log(f"✅ Hipster Shop deployed with {len(lines)} running pods")
            return True
            
        except Exception as e:
            self.errors.append(f"Hipster Shop validation failed: {e}")
            self.log(f"❌ Hipster Shop validation failed: {e}", "ERROR")
            return False
    
    def validate_metrics_collection(self):
        """Validate metrics collection functionality"""
        self.log("Validating metrics collection...")
        try:
            client = PrometheusClient('http://localhost:9090')
            
            # Test basic metrics
            basic_metrics = ['cpu_cores', 'mem_bytes', 'replica_count']
            working_metrics = 0
            
            for metric in basic_metrics:
                try:
                    from datetime import datetime, timedelta
                    end_time = datetime.now()
                    start_time = end_time - timedelta(minutes=1)
                    
                    if metric == 'cpu_cores':
                        result = client._get_cpu_cores('pod=~"frontend.*"', 
                                                    start_time, end_time)
                    elif metric == 'mem_bytes':
                        result = client._get_memory_working_set('pod=~"frontend.*"', 
                                                              start_time, end_time)
                    elif metric == 'replica_count':
                        result = client._get_replica_count('frontend', 'hipster-shop', 
                                                         start_time, end_time)
                    
                    if result is not None:
                        working_metrics += 1
                        
                except Exception as e:
                    self.warnings.append(f"Metric {metric} collection failed: {e}")
            
            # Test complete 12-metric system
            metrics = client.get_comprehensive_metrics('hipster-shop', 'frontend', 1)
            expected_original_metrics = [
                'cpu_cores', 'mem_bytes', 'pod_restarts', 'replica_count', 'node_cpu_util', 'node_mem_util'
            ]
            expected_substitute_metrics = [
                'net_rx_bytes', 'net_tx_bytes', 'network_activity_rate', 'processing_intensity', 
                'service_stability', 'resource_pressure'
            ]
            
            # Check for original metrics
            missing_original = [m for m in expected_original_metrics if m not in metrics]
            missing_substitutes = [m for m in expected_substitute_metrics if m not in metrics]
            
            if missing_original:
                self.warnings.append(f"Missing original metrics: {missing_original}")
            if missing_substitutes:
                self.warnings.append(f"Missing substitute metrics: {missing_substitutes}")
            
            # Count working metrics
            working_original = len([m for m in expected_original_metrics if m in metrics and not metrics[m].empty])
            working_substitutes = len([m for m in expected_substitute_metrics if m in metrics and not metrics[m].empty])
            total_working = working_original + working_substitutes
            
            self.results['metrics_collection'] = {
                'basic_metrics_working': working_metrics,
                'total_metrics_available': len(metrics),
                'expected_metrics': len(expected_original_metrics) + len(expected_substitute_metrics)
            }
            
            self.log(f"✅ Metrics collection: {working_metrics}/{len(basic_metrics)} basic metrics working")
            return True
            
        except Exception as e:
            self.errors.append(f"Metrics collection validation failed: {e}")
            self.log(f"❌ Metrics collection validation failed: {e}", "ERROR")
            return False
    
    def validate_load_generation(self):
        """Validate JMeter load generation"""
        self.log("Validating JMeter load generation...")
        try:
            generator = LoadGenerator()
            
            # Test script creation
            script_path = generator.create_jmeter_script(
                'validation_test', 'localhost', 80, 'browsing', 5
            )
            
            if not os.path.exists(script_path):
                raise Exception("JMeter script not created")
            
            # Test script content
            with open(script_path, 'r') as f:
                content = f.read()
                if 'ThreadGroup' not in content:
                    raise Exception("JMeter script missing ThreadGroup")
            
            # Clean up
            os.remove(script_path)
            
            self.results['load_generation'] = True
            self.log("✅ JMeter load generation is functional")
            return True
            
        except Exception as e:
            self.errors.append(f"Load generation validation failed: {e}")
            self.log(f"❌ Load generation validation failed: {e}", "ERROR")
            return False
    
    def validate_data_acquisition(self):
        """Validate data acquisition pipeline"""
        self.log("Validating data acquisition pipeline...")
        try:
            with open('config/resource-optimized.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            pipeline = DataAcquisitionPipeline(config)
            
            # Test experiment ID generation
            exp_id = pipeline._get_experiment_id('frontend', 'browsing', 1, 10)
            expected = 'frontend_browsing_replicas_1_users_10'
            if exp_id != expected:
                raise Exception(f"Experiment ID mismatch: expected {expected}, got {exp_id}")
            
            # Test experiment completion tracking
            completed = pipeline._get_completed_experiments('frontend')
            if not isinstance(completed, (list, set)):
                raise Exception(f"Completed experiments should return a list or set, got {type(completed)}")
            
            # Test parallel execution method exists
            if not hasattr(pipeline, 'run_parallel_training_experiments'):
                raise Exception("Pipeline missing parallel execution method")
            
            self.results['data_acquisition'] = True
            self.log("✅ Data acquisition pipeline is functional")
            return True
            
        except Exception as e:
            self.errors.append(f"Data acquisition validation failed: {e}")
            self.log(f"❌ Data acquisition validation failed: {e}", "ERROR")
            return False
    
    def validate_cli_commands(self):
        """Validate CLI commands"""
        self.log("Validating CLI commands...")
        try:
            # Test main CLI help
            import subprocess
            result = subprocess.run([
                'python3', '-m', 'src.mora.cli.main', '--help'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                raise Exception(f"CLI help command failed: {result.stderr}")
            
            # Test train command group help
            result = subprocess.run([
                'python3', '-m', 'src.mora.cli.main', 'train', '--help'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                raise Exception(f"Train command help failed: {result.stderr}")
            
            # Test professional ML training command help
            result = subprocess.run([
                'python3', '-m', 'src.mora.cli.main', 'train', 'professional', '--help'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                raise Exception(f"Professional training command help failed: {result.stderr}")
            
            # Test evaluation command help
            result = subprocess.run([
                'python3', '-m', 'src.mora.cli.main', 'train', 'evaluate', '--help'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                raise Exception(f"Evaluation command help failed: {result.stderr}")
            
            self.results['cli_commands'] = True
            self.log("✅ CLI commands are functional (including professional ML pipeline)")
            return True
            
        except Exception as e:
            self.errors.append(f"CLI validation failed: {e}")
            self.log(f"❌ CLI validation failed: {e}", "ERROR")
            return False
    
    def validate_professional_ml_pipeline(self):
        """Validate professional ML pipeline components"""
        self.log("Validating professional ML pipeline components...")
        try:
            # Check if professional ML pipeline files exist
            pipeline_file = Path("train_models/train_professional_ml_pipeline.py")
            evaluator_file = Path("evaluate_models/evaluate_professional_models.py")
            config_file = Path("config/professional_ml_config.json")
            
            if not pipeline_file.exists():
                raise Exception(f"Professional ML pipeline file not found: {pipeline_file}")
            
            if not evaluator_file.exists():
                raise Exception(f"Professional evaluator file not found: {evaluator_file}")
            
            if not config_file.exists():
                raise Exception(f"Professional ML config file not found: {config_file}")
            
            # Test importing the professional ML components
            import sys
            sys.path.insert(0, str(Path.cwd()))
            
            try:
                from train_models.train_professional_ml_pipeline import ProfessionalMLPipeline
                from evaluate_models.evaluate_professional_models import ProfessionalModelEvaluator
            except ImportError as e:
                raise Exception(f"Failed to import professional ML components: {e}")
            
            # Test configuration loading
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            if not isinstance(config, dict):
                raise Exception("Professional ML config is not a valid JSON object")
            
            # Check for required config sections
            required_sections = ['algorithms', 'hyperparameters', 'evaluation']
            for section in required_sections:
                if section not in config:
                    raise Exception(f"Missing required config section: {section}")
            
            self.results['professional_ml_pipeline'] = True
            self.log("✅ Professional ML pipeline components are available")
            return True
            
        except Exception as e:
            self.errors.append(f"Professional ML pipeline validation failed: {e}")
            self.log(f"❌ Professional ML pipeline validation failed: {e}", "ERROR")
            return False
    
    def validate_data_persistence(self):
        self.log("Validating data persistence...")
        try:
            # Ensure training_data directory exists
            os.makedirs('training_data', exist_ok=True)
            
            # Test JSON serialization
            test_data = {
                'experiment_id': 'validation_test',
                'service': 'frontend',
                'scenario': 'browsing',
                'replicas': 1,
                'users': 10,
                'metrics': {
                    'cpu_cores': 'test_data.csv',
                    'mem_bytes': 'test_data.csv'
                }
            }
            
            test_file = 'training_data/validation_test.json'
            with open(test_file, 'w') as f:
                json.dump(test_data, f, indent=2, default=str)
            
            if not os.path.exists(test_file):
                raise Exception("Data file not created")
            
            # Test JSON loading
            with open(test_file, 'r') as f:
                loaded_data = json.load(f)
                if loaded_data['experiment_id'] != 'validation_test':
                    raise Exception("Data loading failed")
            
            # Clean up
            os.remove(test_file)
            
            self.results['data_persistence'] = True
            self.log("✅ Data persistence is functional")
            return True
            
        except Exception as e:
            self.errors.append(f"Data persistence validation failed: {e}")
            self.log(f"❌ Data persistence validation failed: {e}", "ERROR")
            return False
    
    def run_validation(self):
        """Run complete system validation"""
        self.log("🚀 Starting comprehensive system validation...")
        self.log("=" * 60)
        
        validations = [
            ("Minikube Cluster", self.validate_minikube),
            ("Prometheus Connectivity", self.validate_prometheus),
            ("Hipster Shop Deployment", self.validate_hipster_shop),
            ("Metrics Collection", self.validate_metrics_collection),
            ("Load Generation", self.validate_load_generation),
            ("Data Acquisition", self.validate_data_acquisition),
            ("CLI Commands", self.validate_cli_commands),
            ("Professional ML Pipeline", self.validate_professional_ml_pipeline),
            ("Data Persistence", self.validate_data_persistence),
        ]
        
        passed = 0
        total = len(validations)
        
        for name, validation_func in validations:
            self.log(f"\n--- Validating {name} ---")
            if validation_func():
                passed += 1
            time.sleep(1)  # Brief pause between validations
        
        # Summary
        self.log("\n" + "=" * 60)
        self.log("📊 VALIDATION SUMMARY")
        self.log("=" * 60)
        
        self.log(f"✅ Passed: {passed}/{total}")
        self.log(f"❌ Failed: {total - passed}/{total}")
        
        if self.errors:
            self.log(f"\n🚨 ERRORS ({len(self.errors)}):")
            for error in self.errors:
                self.log(f"  - {error}", "ERROR")
        
        if self.warnings:
            self.log(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                self.log(f"  - {warning}", "WARNING")
        
        # Final recommendation
        if passed == total and not self.errors:
            self.log("\n🎉 SYSTEM READY FOR DATA COLLECTION!")
            self.log("All components are functional and ready for high-quality dataset generation.")
            return True
        elif passed >= total * 0.8:  # 80% pass rate
            self.log("\n⚠️  SYSTEM MOSTLY READY")
            self.log("Most components are functional, but some issues need attention.")
            return False
        else:
            self.log("\n❌ SYSTEM NOT READY")
            self.log("Multiple critical issues need to be resolved before data collection.")
            return False


def main():
    """Run system validation"""
    validator = SystemValidator()
    success = validator.run_validation()
    
    if success:
        print("\n" + "=" * 60)
        print("🎯 RECOMMENDATION: Proceed with data collection")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("🛑 RECOMMENDATION: Fix issues before data collection")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
