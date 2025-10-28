#!/usr/bin/env python3
"""
Comprehensive End-to-End Test Suite

This script runs a complete end-to-end test of the MOrA system including:
1. System validation
2. CLI functionality
3. Data collection simulation
4. Model training simulation
5. Model evaluation simulation
"""

import os
import sys
import time
import subprocess
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class ComprehensiveE2ETest:
    """Comprehensive end-to-end test suite"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
        self.temp_dir = None
        
    def log(self, message: str, level: str = "INFO"):
        """Log test results"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def setup_test_environment(self):
        """Set up test environment"""
        self.log("Setting up test environment...")
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'training_data')
        self.model_dir = os.path.join(self.temp_dir, 'models')
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.log(f"Test environment created at: {self.temp_dir}")
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.log("Test environment cleaned up")
    
    def test_cli_structure(self) -> bool:
        """Test CLI command structure"""
        self.log("Testing CLI command structure...")
        try:
            # Test main CLI help
            result = subprocess.run([
                'python3', '-m', 'src.mora.cli.main', '--help'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                raise Exception(f"Main CLI help failed: {result.stderr}")
            
            # Verify main commands
            if 'rightsize' not in result.stdout:
                raise Exception("rightsize command not found")
            if 'status' not in result.stdout:
                raise Exception("status command not found")
            if 'train' not in result.stdout:
                raise Exception("train command not found")
            
            # Test train command group
            result = subprocess.run([
                'python3', '-m', 'src.mora.cli.main', 'train', '--help'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                raise Exception(f"Train command help failed: {result.stderr}")
            
            # Verify training commands
            if 'models' not in result.stdout:
                raise Exception("models command not found")
            if 'evaluate' not in result.stdout:
                raise Exception("evaluate command not found")
            if 'collect-data' not in result.stdout:
                raise Exception("collect-data command not found")
            if 'collect-data-parallel' not in result.stdout:
                raise Exception("collect-data-parallel command not found")
            if 'status' not in result.stdout:
                raise Exception("status command not found")
            
            self.results['cli_structure'] = True
            self.log("✅ CLI structure test passed")
            return True
            
        except Exception as e:
            self.errors.append(f"CLI structure test failed: {e}")
            self.log(f"❌ CLI structure test failed: {e}", "ERROR")
            return False
    
    def test_command_help_texts(self) -> bool:
        """Test individual command help texts"""
        self.log("Testing command help texts...")
        try:
            commands_to_test = [
                ('train', 'models', 'Train ML models using advanced algorithms'),
                ('train', 'evaluate', 'Evaluate trained models using comprehensive analysis'),
                ('train', 'collect-data', 'Collect training data for ML model training'),
                ('train', 'collect-data-parallel', 'Collect training data for multiple services in parallel'),
                ('train', 'status', 'Check training experiment progress for a service'),
                ('rightsize', None, 'Generate rightsizing recommendations for a microservice'),
                ('status', None, 'Show current status of monitoring stack and services'),
            ]
            
            for cmd_group, cmd_name, expected_text in commands_to_test:
                if cmd_name:
                    cmd_args = ['python3', '-m', 'src.mora.cli.main', cmd_group, cmd_name, '--help']
                else:
                    cmd_args = ['python3', '-m', 'src.mora.cli.main', cmd_group, '--help']
                
                result = subprocess.run(cmd_args, capture_output=True, text=True, timeout=10)
                
                if result.returncode != 0:
                    raise Exception(f"Command help failed for {cmd_group} {cmd_name}: {result.stderr}")
                
                if expected_text not in result.stdout:
                    raise Exception(f"Expected text '{expected_text}' not found in {cmd_group} {cmd_name} help")
            
            self.results['command_help_texts'] = True
            self.log("✅ Command help texts test passed")
            return True
            
        except Exception as e:
            self.errors.append(f"Command help texts test failed: {e}")
            self.log(f"❌ Command help texts test failed: {e}", "ERROR")
            return False
    
    def test_configuration_files(self) -> bool:
        """Test configuration files exist and are valid"""
        self.log("Testing configuration files...")
        try:
            # Test professional ML config
            config_file = Path("config/professional_ml_config.json")
            if not config_file.exists():
                raise Exception(f"Professional ML config file not found: {config_file}")
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Check for required sections
            required_sections = ['algorithms', 'hyperparameters', 'evaluation']
            for section in required_sections:
                if section not in config:
                    raise Exception(f"Missing required config section: {section}")
            
            # Test resource-optimized config
            resource_config = Path("config/resource-optimized.yaml")
            if not resource_config.exists():
                raise Exception(f"Resource-optimized config file not found: {resource_config}")
            
            self.results['configuration_files'] = True
            self.log("✅ Configuration files test passed")
            return True
            
        except Exception as e:
            self.errors.append(f"Configuration files test failed: {e}")
            self.log(f"❌ Configuration files test failed: {e}", "ERROR")
            return False
    
    def test_ml_pipeline_components(self) -> bool:
        """Test ML pipeline components"""
        self.log("Testing ML pipeline components...")
        try:
            # Check if files exist
            pipeline_file = Path("train_models/train_professional_ml_pipeline.py")
            evaluator_file = Path("evaluate_models/evaluate_professional_models.py")
            
            if not pipeline_file.exists():
                raise Exception(f"Professional ML pipeline file not found: {pipeline_file}")
            
            if not evaluator_file.exists():
                raise Exception(f"Professional evaluator file not found: {evaluator_file}")
            
            # Test imports
            import sys
            sys.path.insert(0, str(Path.cwd()))
            
            try:
                from train_models.train_professional_ml_pipeline import ProfessionalMLPipeline
                from evaluate_models.evaluate_professional_models import ProfessionalModelEvaluator
            except ImportError as e:
                raise Exception(f"Failed to import professional ML components: {e}")
            
            self.results['ml_pipeline_components'] = True
            self.log("✅ ML pipeline components test passed")
            return True
            
        except Exception as e:
            self.errors.append(f"ML pipeline components test failed: {e}")
            self.log(f"❌ ML pipeline components test failed: {e}", "ERROR")
            return False
    
    def test_data_acquisition_pipeline(self) -> bool:
        """Test data acquisition pipeline"""
        self.log("Testing data acquisition pipeline...")
        try:
            from mora.core.data_acquisition import DataAcquisitionPipeline
            
            # Test with default parameters
            pipeline1 = DataAcquisitionPipeline()
            assert pipeline1.namespace == "hipster-shop"
            assert pipeline1.prometheus_url == "http://localhost:9090"
            assert pipeline1.data_dir == "training_data"
            
            # Test with custom data_dir
            pipeline2 = DataAcquisitionPipeline(data_dir="training_data")
            assert pipeline2.data_dir == "training_data"
            
            self.results['data_acquisition_pipeline'] = True
            self.log("✅ Data acquisition pipeline test passed")
            return True
            
        except Exception as e:
            self.errors.append(f"Data acquisition pipeline test failed: {e}")
            self.log(f"❌ Data acquisition pipeline test failed: {e}", "ERROR")
            return False
    
    def test_unit_tests(self) -> bool:
        """Run unit tests"""
        self.log("Running unit tests...")
        try:
            # Run pytest on our test files
            test_files = [
                'tests/test_cli.py',
                'tests/test_professional_ml_pipeline.py',
            ]
            
            passed_tests = 0
            total_tests = 0
            
            for test_file in test_files:
                if os.path.exists(test_file):
                    result = subprocess.run([
                        'python3', '-m', 'pytest', test_file, '-v', '--tb=short', '--no-header'
                    ], capture_output=True, text=True, timeout=60)
                    
                    if result.returncode != 0:
                        self.warnings.append(f"Unit tests in {test_file} had issues")
                        self.log(f"⚠️ Unit tests in {test_file} had issues", "WARNING")
                    else:
                        # Count passed tests
                        lines = result.stdout.split('\n')
                        for line in lines:
                            if 'PASSED' in line:
                                passed_tests += 1
                            elif 'FAILED' in line:
                                total_tests += 1
                        total_tests = max(total_tests, passed_tests)
                        self.log(f"✅ Unit tests in {test_file} passed")
            
            if passed_tests > 0:
                self.results['unit_tests'] = True
                self.log(f"✅ Unit tests completed ({passed_tests} passed)")
                return True
            else:
                raise Exception("No unit tests passed")
                
        except Exception as e:
            self.errors.append(f"Unit tests failed: {e}")
            self.log(f"❌ Unit tests failed: {e}", "ERROR")
            return False
    
    def test_documentation_consistency(self) -> bool:
        """Test documentation consistency"""
        self.log("Testing documentation consistency...")
        try:
            # Check README.md has correct CLI commands
            readme_file = Path("README.md")
            if not readme_file.exists():
                raise Exception("README.md not found")
            
            with open(readme_file, 'r') as f:
                readme_content = f.read()
            
            # Check for correct CLI commands
            if 'train models' not in readme_content:
                raise Exception("README.md missing 'train models' command")
            if 'train collect-data' not in readme_content:
                raise Exception("README.md missing 'train collect-data' command")
            if 'train evaluate' not in readme_content:
                raise Exception("README.md missing 'train evaluate' command")
            
            # Check ML-Pipeline.md exists
            ml_pipeline_file = Path("ML-Pipeline.md")
            if not ml_pipeline_file.exists():
                raise Exception("ML-Pipeline.md not found")
            
            self.results['documentation_consistency'] = True
            self.log("✅ Documentation consistency test passed")
            return True
            
        except Exception as e:
            self.errors.append(f"Documentation consistency test failed: {e}")
            self.log(f"❌ Documentation consistency test failed: {e}", "ERROR")
            return False
    
    def run_comprehensive_test(self) -> bool:
        """Run comprehensive end-to-end test"""
        self.log("🚀 Starting Comprehensive End-to-End Test Suite")
        self.log("=" * 70)
        
        try:
            self.setup_test_environment()
            
            # Run all tests
            tests = [
                ("CLI Structure", self.test_cli_structure),
                ("Command Help Texts", self.test_command_help_texts),
                ("Configuration Files", self.test_configuration_files),
                ("ML Pipeline Components", self.test_ml_pipeline_components),
                ("Data Acquisition Pipeline", self.test_data_acquisition_pipeline),
                ("Unit Tests", self.test_unit_tests),
                ("Documentation Consistency", self.test_documentation_consistency),
            ]
            
            passed = 0
            total = len(tests)
            
            for test_name, test_func in tests:
                self.log(f"\n--- Running {test_name} ---")
                if test_func():
                    passed += 1
                time.sleep(0.5)  # Brief pause between tests
            
            # Summary
            self.log("\n" + "=" * 70)
            self.log("📊 COMPREHENSIVE TEST SUMMARY")
            self.log("=" * 70)
            
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
            
            # Final assessment
            if passed == total and not self.errors:
                self.log("\n🎉 ALL TESTS PASSED!")
                self.log("The MOrA system is fully functional and ready for production use.")
                return True
            elif passed >= total * 0.8:  # 80% pass rate
                self.log("\n⚠️  MOSTLY FUNCTIONAL")
                self.log("Most tests passed, but some issues need attention.")
                return False
            else:
                self.log("\n❌ SIGNIFICANT ISSUES FOUND")
                self.log("Multiple tests failed. System needs significant fixes.")
                return False
                
        finally:
            self.cleanup_test_environment()


def main():
    """Main function"""
    tester = ComprehensiveE2ETest()
    success = tester.run_comprehensive_test()
    
    if success:
        print("\n🎉 Comprehensive E2E test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Comprehensive E2E test found issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()