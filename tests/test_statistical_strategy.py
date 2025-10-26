"""
Tests for statistical rightsizing strategy
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from src.mora.core.statistical_strategy import StatisticalRightsizer


class TestStatisticalRightsizer:
    """Test cases for statistical rightsizing strategy"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.rightsizer = StatisticalRightsizer(cpu_percentile=95.0, memory_buffer_percentage=15.0)
    
    def test_initialization(self):
        """Test rightsizer initialization"""
        assert self.rightsizer.cpu_percentile == 95.0
        assert self.rightsizer.memory_buffer_percentage == 15.0
    
    def test_parse_cpu_value(self):
        """Test CPU value parsing"""
        # Test millicores
        assert self.rightsizer.parse_cpu_value("100m") == 0.1
        assert self.rightsizer.parse_cpu_value("500m") == 0.5
        
        # Test cores
        assert self.rightsizer.parse_cpu_value("1") == 1.0
        assert self.rightsizer.parse_cpu_value("2.5") == 2.5
        
        # Test edge cases
        assert self.rightsizer.parse_cpu_value("Unknown") == 0.0
        assert self.rightsizer.parse_cpu_value("") == 0.0
        assert self.rightsizer.parse_cpu_value(None) == 0.0
    
    def test_parse_memory_value(self):
        """Test memory value parsing"""
        # Test KiB
        assert self.rightsizer.parse_memory_value("1024Ki") == 1024 * 1024
        
        # Test MiB
        assert self.rightsizer.parse_memory_value("128Mi") == 128 * 1024 * 1024
        
        # Test GiB
        assert self.rightsizer.parse_memory_value("1Gi") == 1024 * 1024 * 1024
        
        # Test edge cases
        assert self.rightsizer.parse_memory_value("Unknown") == 0
        assert self.rightsizer.parse_memory_value("") == 0
        assert self.rightsizer.parse_memory_value(None) == 0
    
    def test_format_cpu_value(self):
        """Test CPU value formatting"""
        # Test millicores for small values
        assert self.rightsizer.format_cpu_value(0.1) == "100m"
        assert self.rightsizer.format_cpu_value(0.5) == "500m"
        
        # Test cores for larger values
        result = self.rightsizer.format_cpu_value(1.0)
        assert result in ["1", "1.0"]
        
        result = self.rightsizer.format_cpu_value(2.5)
        assert result in ["2.5", "2.500"]
    
    def test_format_memory_value(self):
        """Test memory value formatting"""
        # Test GiB formatting
        result = self.rightsizer.format_memory_value(1024 * 1024 * 1024)
        assert "Gi" in result
        
        # Test MiB formatting
        result = self.rightsizer.format_memory_value(128 * 1024 * 1024)
        assert "Mi" in result
        
        # Test KiB formatting
        result = self.rightsizer.format_memory_value(1024)
        assert "Ki" in result
    
    def test_analyze_cpu_usage_with_data(self):
        """Test CPU usage analysis with valid data"""
        # Create test DataFrame
        timestamps = pd.date_range('2023-01-01', periods=100, freq='1min')
        cpu_values = np.random.uniform(0.1, 0.8, 100)
        
        cpu_df = pd.DataFrame({
            'timestamp': timestamps,
            '__value': cpu_values
        })
        
        result = self.rightsizer.analyze_cpu_usage(cpu_df)
        
        assert result['has_data'] == True
        assert 'recommended_requests' in result
        assert 'current_usage_stats' in result
        assert result['current_usage_stats']['count'] == 100
    
    def test_analyze_cpu_usage_empty_data(self):
        """Test CPU usage analysis with no data"""
        empty_df = pd.DataFrame()
        
        result = self.rightsizer.analyze_cpu_usage(empty_df)
        
        assert result['has_data'] == False
        assert result['recommended_requests'] == "100m"  # Default minimum
        assert result['current_usage_stats'] == {}
    
    def test_analyze_memory_usage_with_data(self):
        """Test memory usage analysis with valid data"""
        # Create test DataFrame
        timestamps = pd.date_range('2023-01-01', periods=100, freq='1min')
        memory_values = np.random.uniform(100 * 1024 * 1024, 500 * 1024 * 1024, 100)
        
        memory_df = pd.DataFrame({
            'timestamp': timestamps,
            '__value': memory_values
        })
        
        result = self.rightsizer.analyze_memory_usage(memory_df)
        
        assert result['has_data'] == True
        assert 'recommended_requests' in result
        assert 'max_usage_bytes' in result
        assert result['current_usage_stats']['count'] == 100
        assert result['buffer_percentage'] == 15.0
    
    def test_analyze_memory_usage_empty_data(self):
        """Test memory usage analysis with no data"""
        empty_df = pd.DataFrame()
        
        result = self.rightsizer.analyze_memory_usage(empty_df)
        
        assert result['has_data'] == False
        assert "Mi" in result['recommended_requests']  # Should contain default memory format
    
    def test_generate_recommendations(self):
        """Test recommendation generation"""
        # Mock service data
        service_data = {
            'service_name': 'test-service',
            'namespace': 'default',
            'deployment': {
                'name': 'test-service',
                'containers': [{
                    'name': 'test-container',
                    'resources': {
                        'requests': {'cpu': '100m', 'memory': '128Mi'}
                    }
                }]
            },
            'metrics': {
                'cpu': pd.DataFrame({'__value': [0.1, 0.2, 0.3]}),
                'memory': pd.DataFrame({'__value': [100 * 1024 * 1024, 150 * 1024 * 1024]})
            }
        }
        
        recommendations = self.rightsizer.generate_recommendations(service_data)
        
        assert len(recommendations) == 1
        rec = recommendations[0]
        
        assert rec['service_name'] == 'test-service'
        assert rec['container_name'] == 'test-container'
        assert rec['strategy'] == 'statistical'
        assert 'current_requests' in rec
        assert 'recommended_requests' in rec
        assert 'analysis' in rec
    
    def test_generate_recommendations_no_deployment(self):
        """Test recommendation generation with no deployment data"""
        service_data = {
            'service_name': 'test-service',
            'namespace': 'default'
        }
        
        recommendations = self.rightsizer.generate_recommendations(service_data)
        
        assert recommendations == []
    
    def test_validate_recommendations(self):
        """Test recommendation validation"""
        recommendations = [
            {
                'container_name': 'test-container',
                'recommended_requests': {
                    'cpu': '500m',
                    'memory': '256Mi'
                }
            }
        ]
        
        validation = self.rightsizer.validate_recommendations(recommendations)
        
        assert validation['is_valid'] == True
        assert 'warnings' in validation
        assert 'errors' in validation
    
    def test_validate_recommendations_low_cpu_warning(self):
        """Test validation warning for very low CPU recommendation"""
        recommendations = [
            {
                'container_name': 'test-container',
                'recommended_requests': {
                    'cpu': '5m',  # Very low
                    'memory': '256Mi'
                }
            }
        ]
        
        validation = self.rightsizer.validate_recommendations(recommendations)
        
        assert validation['is_valid'] == True
        assert len(validation['warnings']) > 0
        assert 'low CPU recommendation' in validation['warnings'][0]


class TestStatisticalRightsizerIntegration:
    """Integration tests for statistical rightsizer"""
    
    def setup_method(self):
        """Set up test fixtures for integration tests"""
        self.rightsizer = StatisticalRightsizer()
    
    def test_end_to_end_analysis(self):
        """Test complete end-to-end analysis workflow"""
        # Create realistic test data
        np.random.seed(42)  # For reproducible tests
        
        # CPU data - some variation over time
        cpu_data = np.random.normal(0.3, 0.1, 1440)  # 24 hours of minute data
        cpu_data = np.clip(cpu_data, 0.05, 0.9)  # Clamp to realistic range
        
        cpu_df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=1440, freq='1min'),
            '__value': cpu_data
        })
        
        # Memory data - more stable with some spikes
        memory_data = np.random.normal(200 * 1024 * 1024, 50 * 1024 * 1024, 1440)
        memory_data = np.clip(memory_data, 100 * 1024 * 1024, 500 * 1024 * 1024)
        
        memory_df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=1440, freq='1min'),
            '__value': memory_data
        })
        
        # Analyze both
        cpu_result = self.rightsizer.analyze_cpu_usage(cpu_df)
        memory_result = self.rightsizer.analyze_memory_usage(memory_df)
        
        # Verify results are reasonable
        assert cpu_result['has_data'] == True
        assert memory_result['has_data'] == True
        
        # CPU recommendation should be reasonable (between 0.1 and 1.0 cores for this data)
        cpu_recommended = self.rightsizer.parse_cpu_value(cpu_result['recommended_requests'])
        assert 0.1 <= cpu_recommended <= 1.0
        
        # Memory recommendation should include buffer
        memory_recommended = self.rightsizer.parse_memory_value(memory_result['recommended_requests'])
        max_usage = memory_result['max_usage_bytes']
        buffer_factor = 1.15  # 15% buffer
        expected_min = max_usage * buffer_factor
        assert memory_recommended >= expected_min * 0.9  # Allow some tolerance
