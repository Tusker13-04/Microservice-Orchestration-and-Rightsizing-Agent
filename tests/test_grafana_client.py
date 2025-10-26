"""
Tests for Grafana client integration
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import requests

try:
    from src.mora.monitoring.grafana_client import GrafanaClient
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.mora.monitoring.grafana_client import GrafanaClient


class TestGrafanaClient:
    """Test cases for GrafanaClient"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.client = GrafanaClient(
            grafana_url="http://test-grafana:4000",
            admin_user="admin",
            admin_password="test-password",
            timeout=30
        )
    
    @patch('requests.Session.get')
    def test_test_connection_success(self, mock_get):
        """Test successful connection to Grafana"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.client.test_connection()
        assert result is True
    
    @patch('requests.Session.get')
    def test_test_connection_failure(self, mock_get):
        """Test connection failure to Grafana"""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        result = self.client.test_connection()
        assert result is False
    
    @patch('requests.Session.get')
    def test_get_dashboard_success(self, mock_get):
        """Test successful dashboard retrieval"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'dashboard': {'title': 'Test Dashboard', 'uid': 'test-uid'},
            'meta': {'url': '/d/test-uid'}
        }
        mock_get.return_value = mock_response
        
        result = self.client.get_dashboard('test-uid')
        assert result is not None
        assert result['dashboard']['title'] == 'Test Dashboard'
    
    @patch('requests.Session.get')
    def test_list_dashboards(self, mock_get):
        """Test listing dashboards"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {'title': 'Dashboard 1', 'uid': 'uid-1'},
            {'title': 'Dashboard 2', 'uid': 'uid-2'}
        ]
        mock_get.return_value = mock_response
        
        result = self.client.list_dashboards()
        assert len(result) == 2
        assert result[0]['title'] == 'Dashboard 1'
    
    @patch('requests.Session.get')
    def test_get_data_source(self, mock_get):
        """Test getting Prometheus data source"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'id': 1,
            'name': 'Prometheus',
            'type': 'prometheus',
            'url': 'http://prometheus:9090'
        }
        mock_get.return_value = mock_response
        
        result = self.client.get_data_source('Prometheus')
        assert result is not None
        assert result['name'] == 'Prometheus'
        assert result['type'] == 'prometheus'
    
    @patch('requests.Session.post')
    def test_create_mora_dashboard(self, mock_post):
        """Test MOrA dashboard creation"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'uid': 'mora-test-uid',
            'url': '/d/mora-test-uid'
        }
        mock_post.return_value = mock_response
        
        result = self.client.create_mora_dashboard('test-namespace')
        assert result == 'mora-test-uid'
    
    def test_get_dashboard_url(self):
        """Test dashboard URL generation"""
        url = self.client.get_dashboard_url('test-uid')
        expected = 'http://test-grafana:4000/d/test-uid'
        assert url == expected
    
    @patch('requests.Session.post')
    def test_verify_prometheus_datasource_success(self, mock_post):
        """Test successful Prometheus data source verification"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': {'result': []}}
        mock_post.return_value = mock_response
        
        result = self.client.verify_prometheus_datasource()
        assert result is True
    
    @patch('requests.Session.post')
    def test_verify_prometheus_datasource_failure(self, mock_post):
        """Test Prometheus data source verification failure"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response
        
        result = self.client.verify_prometheus_datasource()
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__])
