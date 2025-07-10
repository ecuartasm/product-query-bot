import pytest
from fastapi.testclient import TestClient
import tempfile
import shutil
import json
import os
from unittest.mock import patch, Mock

# Import the app
from app import app, Config

@pytest.fixture
def client():
    """Create a test client"""
    return TestClient(app)

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_data():
    """Sample product data for testing"""
    return [
        {
            "text": "Product: Test Jacket Description: A waterproof jacket for outdoor activities.",
            "QA": "Q: Is it waterproof? A: Yes, it's fully waterproof."
        },
        {
            "text": "Product: Test Shoes Description: Comfortable running shoes for daily use.",
            "QA": "Q: Are they comfortable? A: Yes, very comfortable for running."
        }
    ]

class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_root_endpoint(self, client):
        """Test the root endpoint returns HTML"""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Product Query Bot" in response.text
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "ready" in data
        assert "database" in data
        assert "timestamp" in data
    
    def test_config_endpoint(self, client):
        """Test configuration endpoint"""
        response = client.get("/config")
        assert response.status_code == 200
        
        data = response.json()
        expected_keys = [
            "vector_db_persist_dir", "collection_name", "embedding_model",
            "top_k", "similarity_threshold", "max_context_length"
        ]
        for key in expected_keys:
            assert key in data
    
    def test_db_info_endpoint(self, client):
        """Test database info endpoint"""
        response = client.get("/db-info")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        # Note: Database might not be initialized in test environment
    
    @patch('app.multi_agent_system')
    def test_web_query_endpoint_success(self, mock_multi_agent, client):
        """Test web query endpoint with successful response"""
        # Mock the multi-agent system
        mock_result = {
            'user_id': 'test_user',
            'query': 'test query',
            'answer': 'Test answer',
            'sources': [],
            'confidence': 'high',
            'timestamp': '2024-01-01T00:00:00'
        }
        
        mock_multi_agent.process_query.return_value = mock_result
        app.multi_agent_system = mock_multi_agent
        
        response = client.post(
            "/web-query",
            data={"user_id": "test_user", "query": "test query"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "result" in data
    
    def test_web_query_endpoint_no_system(self, client):
        """Test web query endpoint when system is not initialized"""
        # Ensure multi_agent_system is None
        app.multi_agent_system = None
        
        response = client.post(
            "/web-query",
            data={"user_id": "test_user", "query": "test query"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data

class TestAPIValidation:
    """Test API input validation"""
    
    def test_web_query_missing_fields(self, client):
        """Test web query with missing required fields"""
        # Missing query
        response = client.post("/web-query", data={"user_id": "test_user"})
        assert response.status_code == 422
        
        # Missing user_id
        response = client.post("/web-query", data={"query": "test query"})
        assert response.status_code == 422
    
    def test_web_query_empty_fields(self, client):
        """Test web query with empty fields"""
        response = client.post(
            "/web-query",
            data={"user_id": "", "query": "test query"}
        )
        # Should still work as user_id can be empty (will be auto-generated)
        assert response.status_code == 200

class TestConfigurationOverrides:
    """Test configuration with environment variable overrides"""
    
    @patch.dict(os.environ, {
        'TOP_K': '10',
        'SIMILARITY_THRESHOLD': '2.0',
        'EMBEDDING_MODEL': 'test-model'
    })
    def test_env_var_override(self, client):
        """Test that environment variables override defaults"""
        # Reload the app module to pick up new env vars
        from importlib import reload
        import app
        reload(app)
        
        response = client.get("/config")
        assert response.status_code == 200
        
        data = response.json()
        assert data["top_k"] == 10
        assert data["similarity_threshold"] == 2.0
        assert data["embedding_model"] == "test-model"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])