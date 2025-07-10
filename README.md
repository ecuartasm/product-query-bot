# Product Query Bot 🤖

A Product Query Bot built with RAG (Retrieval-Augmented Generation) pipeline and Multi-Agent Architecture using LangChain, ChromaDB, and FastAPI.

## 🚀 Features

- **RAG Pipeline**: Semantic search with vector embeddings
- **Multi-Agent Architecture**: Retriever and Responder agents using LangGraph
- **REST API**: FastAPI with automatic validation
- **Web Interface**: Interactive chat interface
- **Persistent Storage**: ChromaDB vector database
- **Docker Support**: Full containerization
- **Unit Tests**: Comprehensive test suite

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.11+
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers (HuggingFace)
- **Multi-Agent**: LangGraph
- **Frontend**: HTML/CSS/JavaScript
- **Containerization**: Docker & Docker Compose
- **Testing**: Pytest

## 📋 Requirements

- Python 3.11+
- Docker & Docker Compose
- 4GB+ RAM
- 2GB+ storage

## 🚀 Quick Start with Docker

## How to use:

### Clone the repository

```bash
  git clone https://github.com/yourusername/product-query-bot.git
  cd product-query-bot
```

### To install requirements
- pip install -r requirements.txt

### To install dev-requirements (for testing)
- pip install -r requirements-dev.txt

### To run the app:
- python app.py

### Run all tests
- python -m pytest test/ -v

### Run with coverage
- python -m pytest test/ --cov=app -v

### To build the Docker:
- docker build -t product-query-bot .

### To run the docker:
- docker-compose up

### Enjoy!


## Sample Image
![product_query_bot](app_screenshot.png)


### 1. Clone the repository
```bash
git clone https://github.com/yourusername/product-query-bot.git
cd product-query-bot
```


## 🔄 **Complete Execution Flow**

### **Step 1: User Request**
```json
POST /web-query
{"user_id": "user123", "query": "waterproof jacket for skiing"}
```

### **Step 2: Input Validation** (Lines 120-124)
- Pydantic validates JSON structure
- Checks user_id and query are not empty

### **Step 3: Multi-Agent Processing** (Lines 686-710)
- **Retriever Agent** (Lines 641-650):
  - Converts query to embeddings
  - Searches vector database
  - Returns top-k similar documents
- **Responder Agent** (Lines 651-685):
  - Receives retrieved documents
  - Generates grounded answer
  - Calculates confidence score

### **Step 4: Response** 
```json
{
  "success": true,
  "result": {
    "user_id": "user123",
    "query": "waterproof jacket for skiing",
    "answer": "Based on our product catalog: Our new TEK O2 technology makes our four-season waterproof pants even more breathable...",
    "sources": [{"id": 4, "name": "EcoFlex 3L Storm Pants", "relevance": 0.85}],
    "confidence": "high",
    "timestamp": "2024-01-15T14:30:25"
  }
}
```

