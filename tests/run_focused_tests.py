#!/usr/bin/env python3
"""
Focused Test Suite for Updated System

This script runs focused tests on the key components that were updated:
1. CLI functionality
2. Professional ML components
3. Configuration validation
"""

import os
import sys
import time
import subprocess
import tempfile
import shutil
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class FocusedTestSuite:
    """Focused test suite for updated components"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log test results"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def test_cli_functionality(self) -> bool:
        """Test CLI functionality"""
        self.log("Testing CLI functionality...")
        try:
            # Test main CLI help
            result = subprocess.run([
                'python3', '-m', 'src.mora.cli.main', '--help'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                raise Exception(f"Main CLI help failed: {result.stderr}")
            
            # Test train command group
            result = subprocess.run([
                'python3', '-m', 'src.mora.cli.main', 'train', '--help'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                raise Exception(f"Train command help failed: {result.stderr}")
            
            # Test models command
            result = subprocess.run([
                'python3', '-m', 'src.mora.cli.main', 'train', 'models', '--help'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                raise Exception(f"Models command help failed: {result.stderr}")
            
            # Test collect-data command
            result = subprocess.run([
                'python3', '-m', 'src.mora.cli.main', 'train', 'collect-data', '--help'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                raise Exception(f"Collect-data command help failed: {result.stderr}")
            
            # Test evaluation command
            result = subprocess.run([
                'python3', '-m', 'src.mora.cli.main', 'train', 'evaluate', '--help'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                raise Exception(f"Evaluation command help failed: {result.stderr}")
            
            self.results['cli_functionality'] = True
            self.log("✅ CLI functionality test passed")
            return True
            
        except Exception as e:
            self.errors.append(f"CLI functionality test failed: {e}")
            self.log(f"❌ CLI functionality test failed: {e}", "ERROR")
            return False
    
    def test_professional_ml_components(self) -> bool:
        """Test professional ML components"""
        self.log("Testing professional ML components...")
        try:
            # Check if files exist
            pipeline_file = Path("train_models/train_professional_ml_pipeline.py")
            evaluator_file = Path("evaluate_models/evaluate_professional_models.py")
            config_file = Path("config/professional_ml_config.json")
            
            if not pipeline_file.exists():
                raise Exception(f"Professional ML pipeline file not found: {pipeline_file}")
            
            if not evaluator_file.exists():
                raise Exception(f"Professional evaluator file not found: {evaluator_file}")
            
            if not config_file.exists():
                raise Exception(f"Professional ML config file not found: {config_file}")
            
            # Test imports
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
            
            # Check for required sections
            required_sections = ['algorithms', 'hyperparameters', 'evaluation']
            for section in required_sections:
                if section not in config:
                    raise Exception(f"Missing required config section: {section}")
            
            self.results['professional_ml_components'] = True
            self.log("✅ Professional ML components test passed")
            return True
            
        except Exception as e:
            self.errors.append(f"Professional ML components test failed: {e}")
            self.log(f"❌ Professional ML components test failed: {e}", "ERROR")
            return False
    
    def test_configuration_validation(self) -> bool:
        """Test configuration validation"""
        self.log("Testing configuration validation...")
        try:
            config_file = Path("config/professional_ml_config.json")
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Validate algorithms section
            if 'algorithms' not in config:
                raise Exception("Missing 'algorithms' section")
            
            algorithms = config['algorithms']
            expected_algorithms = ['lstm', 'prophet', 'xgboost', 'lightgbm', 'random_forest', 'gradient_boosting', 'ensemble']
            for algo in expected_algorithms:
                if algo not in algorithms:
                    raise Exception(f"Missing algorithm: {algo}")
            
            # Validate hyperparameters section
            if 'hyperparameters' not in config:
                raise Exception("Missing 'hyperparameters' section")
            
            hyperparams = config['hyperparameters']
            for algo in expected_algorithms[:-1]:  # Exclude ensemble
                if algo not in hyperparams:
                    raise Exception(f"Missing hyperparameters for algorithm: {algo}")
            
            # Validate evaluation section
            if 'evaluation' not in config:
                raise Exception("Missing 'evaluation' section")
            
            evaluation = config['evaluation']
            required_eval_keys = ['cv_folds', 'scoring_metrics', 'test_size']
            for key in required_eval_keys:
                if key not in evaluation:
                    raise Exception(f"Missing evaluation key: {key}")
            
            self.results['configuration_validation'] = True
            self.log("✅ Configuration validation test passed")
            return True
            
        except Exception as e:
            self.errors.append(f"Configuration validation test failed: {e}")
            self.log(f"❌ Configuration validation test failed: {e}", "ERROR")
            return False
    
    def test_data_acquisition_pipeline(self) -> bool:
        """Test data acquisition pipeline constructor"""
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
    
    def run_focused_tests(self) -> bool:
        """Run focused test suite"""
        self.log("🚀 Starting Focused Test Suite")
        self.log("=" * 50)
        
        # Run all tests
        tests = [
            ("CLI Functionality", self.test_cli_functionality),
            ("Professional ML Components", self.test_professional_ml_components),
            ("Configuration Validation", self.test_configuration_validation),
            ("Data Acquisition Pipeline", self.test_data_acquisition_pipeline),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            self.log(f"\n--- Running {test_name} ---")
            if test_func():
                passed += 1
            time.sleep(0.5)  # Brief pause between tests
        
        # Summary
        self.log("\n" + "=" * 50)
        self.log("📊 FOCUSED TEST SUMMARY")
        self.log("=" * 50)
        
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
            self.log("\n🎉 ALL FOCUSED TESTS PASSED!")
            self.log("The updated components are working correctly.")
            return True
        elif passed >= total * 0.8:  # 80% pass rate
            self.log("\n⚠️  MOSTLY FUNCTIONAL")
            self.log("Most tests passed, but some issues need attention.")
            return False
        else:
            self.log("\n❌ SIGNIFICANT ISSUES FOUND")
            self.log("Multiple tests failed. Components need fixes.")
            return False


def main():
    """Main function"""
    tester = FocusedTestSuite()
    success = tester.run_focused_tests()
    
    if success:
        print("\n🎉 Focused test suite completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Focused test suite found issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
