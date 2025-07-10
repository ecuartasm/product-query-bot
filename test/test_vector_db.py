import pytest
import tempfile
import shutil
import json
import os
from unittest.mock import Mock, patch
from app import VectorDatabaseManager, AnswerGenerator, MultiAgentSystem, OutputLogger, Config

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

@pytest.fixture
def sample_json_file(temp_dir, sample_data):
    """Create a sample JSON file for testing"""
    json_file = os.path.join(temp_dir, "test_products.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    return json_file

class TestVectorDatabaseManager:
    """Test cases for VectorDatabaseManager"""
    
    def test_initialization(self, temp_dir):
        """Test VectorDatabaseManager initialization"""
        db_manager = VectorDatabaseManager(
            collection_name="test_collection",
            persist_directory=temp_dir,
            logger=None
        )
        
        assert db_manager.collection_name == "test_collection"
        assert db_manager.persist_directory == temp_dir
        assert db_manager.embeddings is not None
        assert os.path.exists(temp_dir)
    
    def test_database_exists_false(self, temp_dir):
        """Test _database_exists returns False for empty directory"""
        db_manager = VectorDatabaseManager(persist_directory=temp_dir, logger=None)
        assert not db_manager._database_exists()
    
    def test_create_new_database(self, temp_dir, sample_json_file):
        """Test creating a new vector database"""
        db_manager = VectorDatabaseManager(persist_directory=temp_dir, logger=None)
        
        result = db_manager.create_new_database(sample_json_file)
        assert result is True
        assert db_manager.vectorstore is not None
        assert db_manager._database_exists()
    
    def test_search_similar_products(self, temp_dir, sample_json_file):
        """Test searching for similar products"""
        db_manager = VectorDatabaseManager(persist_directory=temp_dir, logger=None)
        db_manager.create_new_database(sample_json_file)
        
        results = db_manager.search_similar_products("waterproof jacket", k=2)
        
        assert isinstance(results, list)
        assert len(results) <= 2
        
        if results:
            result = results[0]
            assert 'id' in result
            assert 'content' in result
            assert 'score' in result
            assert 'relevance' in result
    
    def test_get_database_info(self, temp_dir, sample_json_file):
        """Test getting database information"""
        db_manager = VectorDatabaseManager(persist_directory=temp_dir, logger=None)
        
        # Test uninitialized database
        info = db_manager.get_database_info()
        assert info['status'] == 'not_initialized'
        assert info['count'] == 0
        
        # Test initialized database
        db_manager.create_new_database(sample_json_file)
        info = db_manager.get_database_info()
        assert info['status'] == 'initialized'
        assert info['count'] > 0
        assert 'database_size_mb' in info
        assert 'config' in info

class TestAnswerGenerator:
    """Test cases for AnswerGenerator"""
    
    def test_generate_answer_no_docs(self):
        """Test answer generation with no retrieved documents"""
        generator = AnswerGenerator()
        result = generator.generate_answer("test query", [])
        
        assert result['answer'] == "I don't have information about that product in my database."
        assert result['confidence'] == 'low'
    
    def test_generate_answer_with_docs(self):
        """Test answer generation with retrieved documents"""
        generator = AnswerGenerator()
        
        mock_docs = [
            {
                'content': 'Product: Test Jacket Description: This is a waterproof jacket perfect for outdoor activities. It keeps you dry in rain.',
                'relevance': 0.8,
                'score': 0.2
            }
        ]
        
        result = generator.generate_answer("waterproof jacket", mock_docs)
        
        assert isinstance(result['answer'], str)
        assert result['answer'].startswith("Based on our product catalog:")
        assert result['confidence'] in ['low', 'medium', 'high']
    
    def test_confidence_calculation(self):
        """Test confidence level calculation"""
        generator = AnswerGenerator()
        
        # High relevance docs
        high_relevance_docs = [{'content': 'Test content with waterproof jacket information.', 'relevance': 0.8, 'score': 0.2}]
        result = generator.generate_answer("waterproof jacket", high_relevance_docs)
        assert result['confidence'] == 'high'
        
        # Medium relevance docs
        medium_relevance_docs = [{'content': 'Test content with some jacket information.', 'relevance': 0.5, 'score': 0.5}]
        result = generator.generate_answer("jacket", medium_relevance_docs)
        assert result['confidence'] == 'medium'
        
        # Low relevance docs
        low_relevance_docs = [{'content': 'Test content with minimal information.', 'relevance': 0.2, 'score': 0.8}]
        result = generator.generate_answer("jacket", low_relevance_docs)
        assert result['confidence'] == 'low'

class TestMultiAgentSystem:
    """Test cases for MultiAgentSystem"""
    
    def test_initialization(self, temp_dir, sample_json_file):
        """Test MultiAgentSystem initialization"""
        db_manager = VectorDatabaseManager(persist_directory=temp_dir, logger=None)
        db_manager.create_new_database(sample_json_file)
        
        multi_agent = MultiAgentSystem(db_manager)
        
        assert multi_agent.vector_db == db_manager
        assert multi_agent.answer_generator is not None
        assert multi_agent.workflow is not None
    
    def test_retriever_agent(self, temp_dir, sample_json_file):
        """Test retriever agent functionality"""
        db_manager = VectorDatabaseManager(persist_directory=temp_dir, logger=None)
        db_manager.create_new_database(sample_json_file)
        
        multi_agent = MultiAgentSystem(db_manager)
        
        state = {
            'user_id': 'test_user',
            'query': 'waterproof jacket',
            'retrieved_docs': [],
            'answer': '',
            'sources': [],
            'confidence': 'low',
            'messages': []
        }
        
        result_state = multi_agent._retriever_agent(state)
        
        assert isinstance(result_state['retrieved_docs'], list)
        assert len(result_state['messages']) > 0
        assert 'Retriever Agent:' in result_state['messages'][0]
    
    def test_responder_agent(self, temp_dir, sample_json_file):
        """Test responder agent functionality"""
        db_manager = VectorDatabaseManager(persist_directory=temp_dir, logger=None)
        db_manager.create_new_database(sample_json_file)
        
        multi_agent = MultiAgentSystem(db_manager)
        
        # Mock retrieved documents
        mock_docs = [
            {
                'id': 0,
                'content': 'Product: Test Jacket Description: Waterproof jacket for outdoor use.',
                'relevance': 0.8,
                'score': 0.2
            }
        ]
        
        state = {
            'user_id': 'test_user',
            'query': 'waterproof jacket',
            'retrieved_docs': mock_docs,
            'answer': '',
            'sources': [],
            'confidence': 'low',
            'messages': []
        }
        
        result_state = multi_agent._responder_agent(state)
        
        assert isinstance(result_state['answer'], str)
        assert len(result_state['answer']) > 0
        assert result_state['confidence'] in ['low', 'medium', 'high']
        assert isinstance(result_state['sources'], list)
        assert len(result_state['messages']) > 0
    
    def test_process_query_end_to_end(self, temp_dir, sample_json_file):
        """Test complete query processing"""
        db_manager = VectorDatabaseManager(persist_directory=temp_dir, logger=None)
        db_manager.create_new_database(sample_json_file)
        
        multi_agent = MultiAgentSystem(db_manager)
        
        result = multi_agent.process_query("test_user", "waterproof jacket")
        
        assert result['user_id'] == 'test_user'
        assert result['query'] == 'waterproof jacket'
        assert isinstance(result['answer'], str)
        assert isinstance(result['sources'], list)
        assert result['confidence'] in ['low', 'medium', 'high']
        assert 'timestamp' in result

class TestConfiguration:
    """Test cases for configuration management"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        # Test that Config class has expected defaults
        assert hasattr(Config, 'TOP_K')
        assert hasattr(Config, 'SIMILARITY_THRESHOLD')
        assert hasattr(Config, 'EMBEDDING_MODEL')
        assert hasattr(Config, 'VECTOR_DB_PERSIST_DIR')
    
    @patch.dict(os.environ, {'TOP_K': '10', 'SIMILARITY_THRESHOLD': '1.5'})
    def test_config_from_env(self):
        """Test configuration loading from environment variables"""
        # Reload config with new environment variables
        from importlib import reload
        import app
        reload(app)
        
        assert app.Config.TOP_K == 10
        assert app.Config.SIMILARITY_THRESHOLD == 1.5

class TestOutputLogger:
    """Test cases for OutputLogger"""
    
    def test_logger_initialization(self, temp_dir):
        """Test logger initialization"""
        log_file = os.path.join(temp_dir, "test.log")
        logger = OutputLogger(log_file)
        
        assert logger.log_file == log_file
        assert os.path.exists(log_file)
        assert isinstance(logger.logs, list)
    
    def test_logging_functionality(self, temp_dir):
        """Test basic logging functionality"""
        log_file = os.path.join(temp_dir, "test.log")
        logger = OutputLogger(log_file)
        
        logger.log("Test message", "INFO")
        
        assert len(logger.logs) > 0
        assert "Test message" in logger.logs[-1]
        assert "INFO" in logger.logs[-1]
        
        # Check file content
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Test message" in content
            assert "INFO" in content

# Integration Tests
class TestIntegration:
    """Integration test cases"""
    
    def test_full_pipeline(self, temp_dir, sample_json_file):
        """Test the complete RAG pipeline"""
        # Initialize components
        logger = OutputLogger(os.path.join(temp_dir, "test.log"))
        db_manager = VectorDatabaseManager(persist_directory=temp_dir, logger=logger)
        
        # Create database
        assert db_manager.create_new_database(sample_json_file) is True
        
        # Initialize multi-agent system
        multi_agent = MultiAgentSystem(db_manager, logger)
        
        # Process query
        result = multi_agent.process_query("integration_test", "comfortable shoes")
        
        # Verify result structure
        assert isinstance(result, dict)
        required_keys = ['user_id', 'query', 'answer', 'sources', 'confidence', 'timestamp']
        for key in required_keys:
            assert key in result
        
        # Verify content
        assert result['user_id'] == 'integration_test'
        assert result['query'] == 'comfortable shoes'
        assert len(result['answer']) > 0
        assert isinstance(result['sources'], list)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])