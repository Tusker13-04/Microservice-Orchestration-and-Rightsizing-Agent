#!/usr/bin/env python3
"""
Industry Standards Analysis Tests

Comprehensive tests for industry standards analysis including:
- Model loading
- Performance analysis
- Industry compliance checking
- Report generation
"""

import os
import sys
import pytest
import tempfile
import shutil
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import joblib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestIndustryStandardsAnalyzer:
    """Test the IndustryStandardsAnalyzer class"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.temp_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Create mock trained models
        self.create_mock_models()
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def create_mock_models(self):
        """Create mock trained models for testing"""
        # Create a mock model for frontend service
        mock_model = {
            'pipeline_type': 'lightweight_lstm_prophet',
            'trained_at': '2024-01-01T00:00:00',
            'target_columns': ['cpu_target', 'memory_target', 'replica_target'],
            'feature_columns': ['cpu_cores_value', 'mem_bytes_value'],
            'data_shape': (100, 10),
            'lstm_models': {
                'cpu_target': {
                    'status': 'success',
                    'mse': 0.001,
                    'mae': 0.05,
                    'r2': 0.8,
                    'confidence': 0.85
                },
                'memory_target': {
                    'status': 'success',
                    'mse': 1e10,
                    'mae': 1e7,
                    'r2': -2.0,
                    'confidence': 0.6
                },
                'replica_target': {
                    'status': 'success',
                    'mse': 0.2,
                    'mae': 0.15,
                    'r2': 0.7,
                    'confidence': 0.8
                }
            }
        }
        
        joblib.dump(
            mock_model,
            os.path.join(self.model_dir, 'frontend_lstm_prophet_pipeline.joblib')
        )
    
    def test_analyzer_initialization(self):
        """Test that analyzer initializes correctly"""
        from evaluate_models.industry_standards_analysis import IndustryStandardsAnalyzer
        
        analyzer = IndustryStandardsAnalyzer(models_dir=self.model_dir)
        
        assert analyzer.models_dir == self.model_dir
        assert analyzer.industry_standards is not None
        assert 'cpu_prediction' in analyzer.industry_standards
    
    def test_load_all_models(self):
        """Test loading all trained models"""
        from evaluate_models.industry_standards_analysis import IndustryStandardsAnalyzer
        
        analyzer = IndustryStandardsAnalyzer(models_dir=self.model_dir)
        models = analyzer.load_all_models()
        
        assert len(models) > 0
        assert 'frontend' in models
        assert models['frontend']['pipeline_type'] == 'lightweight_lstm_prophet'
    
    def test_load_all_models_empty_dir(self):
        """Test loading models from empty directory"""
        from evaluate_models.industry_standards_analysis import IndustryStandardsAnalyzer
        
        empty_dir = os.path.join(self.temp_dir, 'empty_models')
        os.makedirs(empty_dir, exist_ok=True)
        
        analyzer = IndustryStandardsAnalyzer(models_dir=empty_dir)
        models = analyzer.load_all_models()
        
        assert len(models) == 0
    
    def test_analyze_model_performance(self):
        """Test analyzing model performance"""
        from evaluate_models.industry_standards_analysis import IndustryStandardsAnalyzer
        
        analyzer = IndustryStandardsAnalyzer(models_dir=self.model_dir)
        models = analyzer.load_all_models()
        
        analysis = analyzer.analyze_model_performance('frontend', models['frontend'])
        
        assert 'service_name' in analysis
        assert 'model_type' in analysis
        assert 'performance_metrics' in analysis
        assert 'industry_compliance' in analysis
        assert 'overall_score' in analysis
        assert isinstance(analysis['overall_score'], float)
    
    def test_industry_compliance_checking(self):
        """Test industry compliance checking"""
        from evaluate_models.industry_standards_analysis import IndustryStandardsAnalyzer
        
        analyzer = IndustryStandardsAnalyzer(models_dir=self.model_dir)
        
        # Mock metrics
        metrics = {
            'mse': 0.001,
            'mae': 0.05,
            'r2': 0.8,
            'confidence': 0.85
        }
        
        # Check compliance
        standards = analyzer.industry_standards['cpu_prediction']
        
        mse_compliant = metrics['mse'] <= standards['mse_threshold']
        mae_compliant = metrics['mae'] <= standards['mae_threshold']
        r2_compliant = metrics['r2'] >= standards['r2_minimum']
        confidence_compliant = metrics['confidence'] >= standards['confidence_minimum']
        
        assert mse_compliant is True
        assert mae_compliant is True
        assert r2_compliant is True
        assert confidence_compliant is True
    
    def test_generate_analysis_report(self):
        """Test generating analysis report"""
        from evaluate_models.industry_standards_analysis import IndustryStandardsAnalyzer
        
        analyzer = IndustryStandardsAnalyzer(models_dir=self.model_dir)
        models = analyzer.load_all_models()
        
        # Analyze models
        for service_name, model_data in models.items():
            analysis = analyzer.analyze_model_performance(service_name, model_data)
            analyzer.services_analyzed.append(service_name)
            analyzer.analysis_results[service_name] = analysis
        
        # Generate report
        report_path = analyzer.generate_report()
        
        assert os.path.exists(report_path)
        assert report_path.endswith('.txt')
    
    def test_run_comprehensive_analysis(self):
        """Test running comprehensive analysis"""
        from evaluate_models.industry_standards_analysis import IndustryStandardsAnalyzer
        
        analyzer = IndustryStandardsAnalyzer(models_dir=self.model_dir)
        analyzer.run_comprehensive_analysis()
        
        assert len(analyzer.services_analyzed) > 0
        assert len(analyzer.analysis_results) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


