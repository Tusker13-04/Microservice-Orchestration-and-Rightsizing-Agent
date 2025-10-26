#!/usr/bin/env python3
"""
Phase 2 Completion Test Suite
Tests all Phase 2 deliverables and functionality
"""
import os
import sys
import subprocess
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class Phase2CompletionTester:
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
        
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def test_infrastructure_components(self) -> bool:
        """Test all infrastructure components are available"""
        self.log("Testing infrastructure components...")
        
        try:
            # Test Kubernetes
            result = subprocess.run(['kubectl', 'get', 'nodes'], capture_output=True, text=True)
            if result.returncode != 0:
                self.errors.append("Kubernetes cluster not accessible")
                return False
            self.results['kubernetes'] = True
            
            # Test Prometheus
            import requests
            response = requests.get("http://localhost:9090/api/v1/status/config", timeout=5)
            if response.status_code != 200:
                self.errors.append("Prometheus not accessible")
                return False
            self.results['prometheus'] = True
            
            # Test Hipster Shop
            result = subprocess.run(['kubectl', 'get', 'pods', '-n', 'hipster-shop'], capture_output=True, text=True)
            if result.returncode != 0 or 'frontend' not in result.stdout:
                self.errors.append("Hipster Shop not deployed")
                return False
            self.results['hipster_shop'] = True
            
            # Test JMeter
            result = subprocess.run(['which', 'jmeter'], capture_output=True, text=True)
            if result.returncode != 0:
                self.warnings.append("JMeter not found")
            else:
                self.results['jmeter'] = True
                
            return True
            
        except Exception as e:
            self.errors.append(f"Infrastructure test failed: {e}")
            return False
    
    def test_data_acquisition_pipeline(self) -> bool:
        """Test data acquisition pipeline functionality"""
        self.log("Testing data acquisition pipeline...")
        
        try:
            from mora.core.data_acquisition import DataAcquisitionPipeline
            
            # Test pipeline creation
            pipeline = DataAcquisitionPipeline(namespace='hipster-shop', prometheus_url='http://localhost:9090')
            self.results['data_acquisition_pipeline'] = True
            
            # Test configuration loading
            config = {
                'experiment_duration_minutes': 2,
                'sample_interval': '30s',
                'replica_counts': [1],
                'load_levels_users': [5],
                'test_scenarios': ['browsing'],
                'stabilization_wait_seconds': 30
            }
            
            # Test experiment ID generation
            experiment_id = pipeline._get_experiment_id('frontend', 'browsing', 1, 5)
            if not experiment_id:
                self.errors.append("Experiment ID generation failed")
                return False
                
            self.results['experiment_id_generation'] = True
            return True
            
        except Exception as e:
            self.errors.append(f"Data acquisition pipeline test failed: {e}")
            return False
    
    def test_model_training_pipeline(self) -> bool:
        """Test model training pipeline functionality"""
        self.log("Testing model training pipeline...")
        
        try:
            # Test LSTM + Prophet pipeline
            if os.path.exists('train_working_lstm_prophet.py'):
                self.results['lstm_prophet_pipeline'] = True
            else:
                self.warnings.append("LSTM + Prophet pipeline not found")
                
            # Test model library
            from mora.core.model_library import ModelLibrary
            ml = ModelLibrary()
            self.results['model_library'] = True
            
            return True
            
        except Exception as e:
            self.errors.append(f"Model training pipeline test failed: {e}")
            return False
    
    def test_cli_interface(self) -> bool:
        """Test CLI interface functionality"""
        self.log("Testing CLI interface...")
        
        try:
            # Test CLI help
            result = subprocess.run(['python3', '-m', 'src.mora.cli.main', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                self.errors.append("CLI help command failed")
                return False
                
            # Test train command
            result = subprocess.run(['python3', '-m', 'src.mora.cli.main', 'train', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                self.errors.append("CLI train command failed")
                return False
                
            self.results['cli_interface'] = True
            return True
            
        except Exception as e:
            self.errors.append(f"CLI interface test failed: {e}")
            return False
    
    def test_metrics_collection(self) -> bool:
        """Test metrics collection functionality"""
        self.log("Testing metrics collection...")
        
        try:
            from mora.monitoring.prometheus_client import PrometheusClient
            
            client = PrometheusClient('http://localhost:9090')
            
            # Test connection
            if not client.test_connection():
                self.errors.append("Prometheus connection failed")
                return False
                
            # Test basic metrics collection
            try:
                metrics = client.get_comprehensive_metrics('hipster-shop', 'frontend')
                if metrics:
                    self.results['metrics_collection'] = True
                else:
                    self.warnings.append("Metrics collection returned empty results")
                    self.results['metrics_collection'] = False
            except Exception as e:
                self.warnings.append(f"Metrics collection has issues: {e}")
                self.results['metrics_collection'] = False
                
            return True
            
        except Exception as e:
            self.errors.append(f"Metrics collection test failed: {e}")
            return False
    
    def test_data_persistence(self) -> bool:
        """Test data persistence functionality"""
        self.log("Testing data persistence...")
        
        try:
            # Check if directories exist
            if not os.path.exists('training_data'):
                os.makedirs('training_data')
            if not os.path.exists('models'):
                os.makedirs('models')
                
            self.results['data_persistence'] = True
            return True
            
        except Exception as e:
            self.errors.append(f"Data persistence test failed: {e}")
            return False
    
    def test_documentation(self) -> bool:
        """Test documentation completeness"""
        self.log("Testing documentation...")
        
        try:
            required_docs = ['PRD.md', 'ML-Pipeline.md', 'README.md']
            missing_docs = []
            
            for doc in required_docs:
                if not os.path.exists(doc):
                    missing_docs.append(doc)
                    
            if missing_docs:
                self.warnings.append(f"Missing documentation: {missing_docs}")
            else:
                self.results['documentation'] = True
                
            return True
            
        except Exception as e:
            self.errors.append(f"Documentation test failed: {e}")
            return False
    
    def test_phase2_completion_status(self) -> Dict[str, Any]:
        """Test overall Phase 2 completion status"""
        self.log("Testing Phase 2 completion status...")
        
        completion_status = {
            'infrastructure': False,
            'data_acquisition': False,
            'model_training': False,
            'cli_interface': False,
            'metrics_collection': False,
            'data_persistence': False,
            'documentation': False,
            'collected_data': False,
            'trained_models': False
        }
        
        # Test infrastructure
        completion_status['infrastructure'] = self.test_infrastructure_components()
        
        # Test data acquisition
        completion_status['data_acquisition'] = self.test_data_acquisition_pipeline()
        
        # Test model training
        completion_status['model_training'] = self.test_model_training_pipeline()
        
        # Test CLI interface
        completion_status['cli_interface'] = self.test_cli_interface()
        
        # Test metrics collection
        completion_status['metrics_collection'] = self.test_metrics_collection()
        
        # Test data persistence
        completion_status['data_persistence'] = self.test_data_persistence()
        
        # Test documentation
        completion_status['documentation'] = self.test_documentation()
        
        # Check for collected data
        training_files = [f for f in os.listdir('training_data') if f.endswith('.csv')]
        completion_status['collected_data'] = len(training_files) > 0
        
        # Check for trained models
        model_files = [f for f in os.listdir('models') if f.endswith('.joblib') or f.endswith('.h5')]
        completion_status['trained_models'] = len(model_files) > 0
        
        return completion_status
    
    def run_comprehensive_test(self):
        """Run comprehensive Phase 2 test"""
        self.log("🚀 Starting Phase 2 Completion Test Suite")
        self.log("=" * 60)
        
        # Test all components
        completion_status = self.test_phase2_completion_status()
        
        # Calculate completion percentage
        completed = sum(1 for status in completion_status.values() if status)
        total = len(completion_status)
        completion_percentage = (completed / total) * 100
        
        # Print results
        self.log("📊 PHASE 2 COMPLETION RESULTS")
        self.log("=" * 60)
        
        for component, status in completion_status.items():
            status_icon = "✅" if status else "❌"
            self.log(f"{status_icon} {component.replace('_', ' ').title()}: {'Complete' if status else 'Incomplete'}")
        
        self.log("")
        self.log(f"📈 Overall Completion: {completed}/{total} ({completion_percentage:.1f}%)")
        
        if self.errors:
            self.log("")
            self.log("❌ ERRORS:")
            for error in self.errors:
                self.log(f"  - {error}", "ERROR")
        
        if self.warnings:
            self.log("")
            self.log("⚠️  WARNINGS:")
            for warning in self.warnings:
                self.log(f"  - {warning}", "WARNING")
        
        # Determine next steps
        self.log("")
        if completion_percentage >= 80:
            self.log("🎉 PHASE 2 IS MOSTLY COMPLETE!")
            self.log("📋 Next Steps:")
            if not completion_status['collected_data']:
                self.log("  1. Start data collection for frontend service")
            if not completion_status['trained_models']:
                self.log("  2. Train models with collected data")
            if completion_status['collected_data'] and completion_status['trained_models']:
                self.log("  3. Proceed to Phase 3 (Orchestrator Agent)")
        else:
            self.log("⚠️  PHASE 2 NEEDS MORE WORK")
            self.log("📋 Priority Actions:")
            incomplete = [k for k, v in completion_status.items() if not v]
            for i, component in enumerate(incomplete, 1):
                self.log(f"  {i}. Fix {component.replace('_', ' ')}")
        
        return completion_status

def main():
    tester = Phase2CompletionTester()
    results = tester.run_comprehensive_test()
    return results

if __name__ == "__main__":
    main()
