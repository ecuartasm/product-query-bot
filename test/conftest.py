import pytest
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the app
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables for testing
os.environ.setdefault("VECTOR_DB_PERSIST_DIR", "./test_vector_db")
os.environ.setdefault("TOP_K", "5")
os.environ.setdefault("SIMILARITY_THRESHOLD", "1.2")
os.environ.setdefault("DATA_FILE_PATH", "data/db_20_items.json")

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test"""
    # Ensure we're in the correct directory
    os.chdir(project_root)
    yield
    # Cleanup after test if needed