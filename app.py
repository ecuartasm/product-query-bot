import json
import os
from typing import List, Dict, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, validator
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from environment variables
class Config:
    # Vector Database Configuration
    VECTOR_DB_PERSIST_DIR = os.getenv("VECTOR_DB_PERSIST_DIR", "./vector_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "product_collection")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Search Configuration
    TOP_K = int(os.getenv("TOP_K", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "1.2"))
    MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "2000"))
    
    # Application Configuration
    LOG_FILE = os.getenv("LOG_FILE", "product_query_bot_log.txt")
    DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "data/db_20_items.json")
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "8000"))
    
    # Answer Generation Configuration
    MAX_ANSWER_SENTENCES = int(os.getenv("MAX_ANSWER_SENTENCES", "3"))
    MIN_SENTENCE_LENGTH = int(os.getenv("MIN_SENTENCE_LENGTH", "20"))

# Output Logger Class
class OutputLogger:
    def __init__(self, log_file: str = None):
        self.log_file = log_file or Config.LOG_FILE
        self.logs = []
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Initialize the log file with header if it doesn't exist"""
        if not os.path.exists(self.log_file):
            header = f"""{'='*80}
PRODUCT QUERY BOT - OUTPUT LOG
{'='*80}
Log file created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Configuration:
- Vector DB Directory: {Config.VECTOR_DB_PERSIST_DIR}
- Collection Name: {Config.COLLECTION_NAME}
- Embedding Model: {Config.EMBEDDING_MODEL}
- Top K: {Config.TOP_K}
- Similarity Threshold: {Config.SIMILARITY_THRESHOLD}
{'='*80}

"""
            try:
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    f.write(header)
                print(f"Log file initialized: {self.log_file}")
            except Exception as e:
                print(f"Error initializing log file: {e}")
        else:
            print(f"Using existing log file: {self.log_file}")
    
    def log(self, message: str, log_type: str = "INFO"):
        """Add a message to the log"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{log_type}] {message}"
        
        # Add to memory
        self.logs.append(log_entry)
        
        # Write to file immediately
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + "\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")
        
        # Also print to console
        print(log_entry)
    
    def log_query_interaction(self, user_id: str, query: str, response: str, sources: List[Dict], confidence: str):
        """Log a complete query interaction"""
        interaction_log = f"""
{'='*60}
QUERY INTERACTION
{'='*60}
User ID: {user_id}
Query: {query}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Response:
{response}

Confidence: {confidence.upper()}

Sources ({len(sources)} found):"""
        
        for i, source in enumerate(sources, 1):
            interaction_log += f"\n  {i}. ID: {source.get('id', 'N/A')} | Name: {source.get('name', 'Unknown')} | Relevance: {source.get('relevance', 0):.1%}"
        
        interaction_log += f"\n{'='*60}\n"
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(interaction_log)
        except Exception as e:
            print(f"Error logging interaction: {e}")

# Pydantic models for API
class QueryRequest(BaseModel):
    user_id: str
    query: str
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not v or not v.strip():
            raise ValueError('user_id cannot be empty')
        return v.strip()
    
    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('query cannot be empty')
        return v.strip()

# State for multi-agent system
class AgentState(TypedDict):
    user_id: str
    query: str
    retrieved_docs: List[Dict[str, Any]]
    answer: str
    sources: List[Dict[str, Any]]
    confidence: str
    messages: Annotated[List[str], add_messages]

# Vector Database Manager with Persistent Storage
class VectorDatabaseManager:
    def __init__(self, collection_name: str = None, persist_directory: str = None, logger: OutputLogger = None):
        self.logger = logger
        self.persist_directory = persist_directory or Config.VECTOR_DB_PERSIST_DIR
        self.collection_name = collection_name or Config.COLLECTION_NAME
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        if self.logger:
            self.logger.log(f"Vector database directory: {self.persist_directory}")
            self.logger.log(f"Using embedding model: {Config.EMBEDDING_MODEL}")
            self.logger.log("Initializing Hugging Face embeddings...")
        else:
            print(f"Vector database directory: {self.persist_directory}")
            print(f"Using embedding model: {Config.EMBEDDING_MODEL}")
            print("Initializing Hugging Face embeddings...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        self.vectorstore = None
        
        if self.logger:
            self.logger.log("Vector database manager initialized!")
        else:
            print("Vector database manager initialized!")
    
    def _database_exists(self) -> bool:
        """Check if the vector database already exists"""
        db_files = [
            os.path.join(self.persist_directory, "chroma.sqlite3"),
            os.path.join(self.persist_directory, "index")
        ]
        return any(os.path.exists(f) for f in db_files) or len(os.listdir(self.persist_directory)) > 0
    
    def load_existing_database(self):
        """Load existing vector database from disk"""
        try:
            if self.logger:
                self.logger.log("Loading existing vector database from disk...")
            else:
                print("Loading existing vector database from disk...")
            
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            # Test the database by getting collection info
            collection = self.vectorstore._collection
            count = collection.count()
            
            success_message = f"Successfully loaded existing vector database with {count} documents"
            if self.logger:
                self.logger.log(success_message)
            else:
                print(success_message)
            
            return True
            
        except Exception as e:
            error_message = f"Error loading existing database: {e}"
            if self.logger:
                self.logger.log(error_message, "ERROR")
            else:
                print(error_message)
            return False
    
    def create_new_database(self, json_file_path: str):
        """Create a new vector database from JSON file"""
        try:
            data = []
            with open(json_file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            
            message = f"Loaded {len(data)} products from {json_file_path}"
            if self.logger:
                self.logger.log(message)
            else:
                print(message)
            
            # Prepare documents
            documents = []
            for i, item in enumerate(data):
                content = f"{item['text']}\n\nQ&A:\n{item['QA']}"
                doc = Document(
                    page_content=content,
                    metadata={
                        "id": i,
                        "source": "product_catalog",
                        "text": item['text'],
                        "qa": item['QA']
                    }
                )
                documents.append(doc)
            
            # Create vector store with persistence
            if self.logger:
                self.logger.log("Creating new vector database with embeddings...")
            else:
                print("Creating new vector database with embeddings...")
            
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            
            # Explicitly persist the database
            if hasattr(self.vectorstore, 'persist'):
                self.vectorstore.persist()
                if self.logger:
                    self.logger.log("Vector database persisted to disk")
                else:
                    print("Vector database persisted to disk")
            
            success_message = f"New vector database created with {len(documents)} documents and saved to {self.persist_directory}"
            if self.logger:
                self.logger.log(success_message)
            else:
                print(success_message)
            
            return True
            
        except Exception as e:
            error_message = f"Error creating new database: {e}"
            if self.logger:
                self.logger.log(error_message, "ERROR")
            else:
                print(error_message)
            return False
    
    def load_and_index_products(self, json_file_path: str = None):
        """Load products from JSON and create/load vector database"""
        json_file_path = json_file_path or Config.DATA_FILE_PATH
        
        # Check if database already exists
        if self._database_exists():
            if self.logger:
                self.logger.log("Existing vector database found. Attempting to load...")
            else:
                print("Existing vector database found. Attempting to load...")
            
            if self.load_existing_database():
                return True
            else:
                if self.logger:
                    self.logger.log("Failed to load existing database. Creating new one...")
                else:
                    print("Failed to load existing database. Creating new one...")
        else:
            if self.logger:
                self.logger.log("No existing database found. Creating new vector database...")
            else:
                print("No existing database found. Creating new vector database...")
        
        # Create new database
        return self.create_new_database(json_file_path)
    
    def search_similar_products(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """Search for similar products and return deduplicated results"""
        if not self.vectorstore:
            return []
        
        k = k or Config.TOP_K
        
        try:
            # Get more results to account for potential duplicates
            raw_results = self.vectorstore.similarity_search_with_score(query, k=k*2)
            
            # Deduplicate based on document ID
            seen_ids = set()
            unique_results = []
            
            for doc, score in raw_results:
                doc_id = doc.metadata.get('id')
                if doc_id not in seen_ids and score <= Config.SIMILARITY_THRESHOLD:
                    seen_ids.add(doc_id)
                    unique_results.append({
                        'id': doc_id,
                        'content': doc.page_content,
                        'score': score,
                        'relevance': 1/(1+score),
                        'metadata': doc.metadata
                    })
            
            search_message = f"Found {len(unique_results)} unique products for query: '{query}'"
            if self.logger:
                self.logger.log(search_message)
            
            return unique_results[:k]
            
        except Exception as e:
            error_message = f"Error searching products: {e}"
            if self.logger:
                self.logger.log(error_message, "ERROR")
            else:
                print(error_message)
            return []
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the current database"""
        if not self.vectorstore:
            return {"status": "not_initialized", "count": 0, "persist_directory": self.persist_directory}
        
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            # Get database file sizes
            db_size = 0
            if os.path.exists(self.persist_directory):
                for root, dirs, files in os.walk(self.persist_directory):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            db_size += os.path.getsize(file_path)
            
            return {
                "status": "initialized",
                "count": count,
                "persist_directory": self.persist_directory,
                "collection_name": self.collection_name,
                "database_size_mb": round(db_size / (1024 * 1024), 2),
                "exists_on_disk": self._database_exists(),
                "config": {
                    "top_k": Config.TOP_K,
                    "similarity_threshold": Config.SIMILARITY_THRESHOLD,
                    "embedding_model": Config.EMBEDDING_MODEL
                }
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e),
                "persist_directory": self.persist_directory
            }

# Simple RAG Answer Generator
class AnswerGenerator:
    def __init__(self, logger: OutputLogger = None):
        self.logger = logger
    
    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer from retrieved documents using keyword-based approach"""
        if not retrieved_docs:
            answer = "I don't have information about that product in my database."
            return {'answer': answer, 'confidence': 'low'}
        
        # Extract sentences from all retrieved documents
        all_sentences = []
        for doc in retrieved_docs:
            content = doc['content']
            sentences = []
            for part in content.split('\n\n'):
                sentences.extend([s.strip() for s in part.split('.') if s.strip() and len(s.strip()) > Config.MIN_SENTENCE_LENGTH])
            all_sentences.extend(sentences)
        
        # Extract query keywords
        stop_words = {'the', 'and', 'or', 'but', 'for', 'with', 'what', 'how', 'when', 'where', 'why', 'who', 'is', 'are', 'can', 'do', 'does'}
        query_words = set([
            word.lower().strip('.,!?;:') for word in query.lower().split() 
            if len(word) > 2 and word.lower() not in stop_words
        ])
        
        # Score sentences based on keyword overlap
        sentence_scores = []
        for sentence in all_sentences:
            sentence_words = set([word.lower().strip('.,!?;:') for word in sentence.split()])
            overlap = len(query_words.intersection(sentence_words))
            if overlap > 0:
                score = overlap / len(sentence_words) * overlap
                sentence_scores.append((sentence, score))
        
        # Sort by score and build answer
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        if not sentence_scores:
            return {'answer': "I found some related products, but I need more specific information to provide a detailed answer.", 'confidence': 'low'}
        
        # Build answer from top sentences
        answer_parts = []
        used_content = set()
        
        for sentence, score in sentence_scores[:Config.MAX_ANSWER_SENTENCES]:
            sentence_key = sentence.lower()[:50]
            if sentence_key not in used_content:
                answer_parts.append(sentence.strip())
                used_content.add(sentence_key)
        
        if answer_parts:
            answer = "Based on our product catalog: " + ". ".join(answer_parts)
            if not answer.endswith('.'):
                answer += "."
            
            # Determine confidence
            avg_relevance = sum(doc['relevance'] for doc in retrieved_docs) / len(retrieved_docs)
            confidence = 'high' if avg_relevance > 0.6 else 'medium' if avg_relevance > 0.4 else 'low'
            
            return {'answer': answer, 'confidence': confidence}
        else:
            return {'answer': "I found some related products, but I need more specific information to provide a detailed answer.", 'confidence': 'low'}

# Multi-Agent System
class MultiAgentSystem:
    def __init__(self, vector_db_manager: VectorDatabaseManager, logger: OutputLogger = None):
        self.vector_db = vector_db_manager
        self.answer_generator = AnswerGenerator(logger)
        self.logger = logger
        self.workflow = self._create_workflow()
    
    def _create_workflow(self):
        """Create the multi-agent workflow using LangGraph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes (agents)
        workflow.add_node("retriever_agent", self._retriever_agent)
        workflow.add_node("responder_agent", self._responder_agent)
        
        # Define the flow
        workflow.set_entry_point("retriever_agent")
        workflow.add_edge("retriever_agent", "responder_agent")
        workflow.add_edge("responder_agent", END)
        
        return workflow.compile()
    
    def _retriever_agent(self, state: AgentState) -> AgentState:
        """Agent responsible for retrieving relevant documents"""
        retrieved_docs = self.vector_db.search_similar_products(state['query'], k=Config.TOP_K)
        state['retrieved_docs'] = retrieved_docs
        state['messages'].append(f"Retriever Agent: Found {len(retrieved_docs)} relevant products")
        return state
    
    def _responder_agent(self, state: AgentState) -> AgentState:
        """Agent responsible for generating the final response"""
        result = self.answer_generator.generate_answer(state['query'], state['retrieved_docs'])
        
        # Prepare sources
        sources = []
        for doc in state['retrieved_docs']:
            product_name = "Unknown Product"
            if "Product:" in doc['content']:
                try:
                    product_name = doc['content'].split("Product:")[1].split("Description:")[0].strip()
                except:
                    pass
            
            sources.append({
                'id': doc['id'],
                'name': product_name,
                'relevance': doc['relevance'],
                'score': doc['score']
            })
        
        state['answer'] = result['answer']
        state['confidence'] = result['confidence']
        state['sources'] = sources
        state['messages'].append(f"Responder Agent: Generated answer with {result['confidence']} confidence")
        
        return state
    
    def process_query(self, user_id: str, query: str) -> Dict[str, Any]:
        """Process a user query through the multi-agent system"""
        initial_state = {
            'user_id': user_id,
            'query': query,
            'retrieved_docs': [],
            'answer': '',
            'sources': [],
            'confidence': 'low',
            'messages': []
        }
        
        final_state = self.workflow.invoke(initial_state)
        
        return {
            'user_id': final_state['user_id'],
            'query': final_state['query'],
            'answer': final_state['answer'],
            'sources': final_state['sources'],
            'confidence': final_state['confidence'],
            'timestamp': datetime.now().isoformat()
        }

# Create templates directory and files
def create_web_files():
    """Create necessary directories and HTML template"""
    os.makedirs("templates", exist_ok=True)
    
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Product Query Bot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px;
        }
        .container {
            max-width: 800px; margin: 0 auto; background: white;
            border-radius: 15px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 30px; text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.1em; opacity: 0.9; }
        .db-info {
            background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px;
            margin-top: 15px; font-size: 0.9em;
        }
        .chat-container { padding: 30px; min-height: 400px; }
        .query-form { margin-bottom: 30px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; }
        input[type="text"], textarea {
            width: 100%; padding: 15px; border: 2px solid #e1e5e9;
            border-radius: 10px; font-size: 16px; transition: border-color 0.3s ease;
        }
        input[type="text"]:focus, textarea:focus { outline: none; border-color: #667eea; }
        textarea { resize: vertical; min-height: 100px; }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 15px 30px; border: none; border-radius: 10px;
            font-size: 16px; font-weight: 600; cursor: pointer; transition: transform 0.2s ease; width: 100%;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn-secondary {
            background: #6c757d; margin-top: 10px; padding: 10px 20px; font-size: 14px;
        }
        .response-container {
            margin-top: 30px; padding: 25px; background: #f8f9fa;
            border-radius: 10px; border-left: 4px solid #667eea;
        }
        .confidence { padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; text-transform: uppercase; }
        .confidence.high { background: #d4edda; color: #155724; }
        .confidence.medium { background: #fff3cd; color: #856404; }
        .confidence.low { background: #f8d7da; color: #721c24; }
        .answer { font-size: 16px; line-height: 1.6; margin-bottom: 20px; color: #333; }
        .sources { margin-top: 20px; }
        .sources h4 { margin-bottom: 10px; color: #667eea; }
        .source-item { background: white; padding: 10px 15px; margin-bottom: 8px; border-radius: 8px; border-left: 3px solid #667eea; }
        .source-name { font-weight: 600; color: #333; }
        .source-relevance { font-size: 14px; color: #666; }
        .timestamp { font-size: 12px; color: #666; text-align: right; margin-top: 15px; }
        .loading { display: none; text-align: center; padding: 20px; }
        .spinner {
            border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%;
            width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 10px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .error { background: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; margin-top: 20px; }
        .success { background: #d4edda; color: #155724; padding: 15px; border-radius: 8px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Product Query Bot</h1>
            <p>Ask me anything about our products!</p>
            <div class="db-info" id="dbInfo">
                💾 Loading database information...
            </div>
        </div>
        <div class="chat-container">
            <form class="query-form" id="queryForm">
                <div class="form-group">
                    <label for="user_id">Your Name (optional):</label>
                    <input type="text" id="user_id" name="user_id" placeholder="Enter your name or leave blank">
                </div>
                <div class="form-group">
                    <label for="query">Your Question:</label>
                    <textarea id="query" name="query" placeholder="Ask about waterproof jackets, shoes, furniture..." required></textarea>
                </div>
                <button type="submit" class="btn">Ask Question 🚀</button>
                <button type="button" class="btn btn-secondary" onclick="refreshDbInfo()">Refresh DB Info 🔄</button>
            </form>
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing your question...</p>
            </div>
            <div id="response-area"></div>
        </div>
    </div>
    <script>
        window.onload = function() { refreshDbInfo(); };
        
        async function refreshDbInfo() {
            try {
                const response = await fetch('/db-info');
                const data = await response.json();
                const dbInfo = document.getElementById('dbInfo');
                
                if (data.status === 'initialized') {
                    dbInfo.innerHTML = `💾 Vector DB: ${data.count} docs | ${data.database_size_mb} MB | Top-K: ${data.config.top_k} | 📁 ${data.persist_directory}`;
                } else {
                    dbInfo.innerHTML = `💾 Database Status: ${data.status}`;
                }
            } catch (error) {
                document.getElementById('dbInfo').innerHTML = '💾 Database info unavailable';
            }
        }
        
        document.getElementById('queryForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const userIdInput = document.getElementById('user_id');
            const queryInput = document.getElementById('query');
            const loading = document.getElementById('loading');
            const responseArea = document.getElementById('response-area');
            const userId = userIdInput.value.trim() || 'anonymous_' + Math.random().toString(36).substr(2, 9);
            const query = queryInput.value.trim();
            if (!query) { alert('Please enter a question!'); return; }
            loading.style.display = 'block';
            responseArea.innerHTML = '';
            try {
                const response = await fetch('/web-query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: `user_id=${encodeURIComponent(userId)}&query=${encodeURIComponent(query)}`
                });
                const data = await response.json();
                loading.style.display = 'none';
                if (data.success) {
                    const result = data.result;
                    const timestamp = new Date(result.timestamp).toLocaleString();
                    let sourcesHtml = '';
                    if (result.sources && result.sources.length > 0) {
                        sourcesHtml = `<div class="sources"><h4>📋 Sources (${result.sources.length} products found):</h4>
                        ${result.sources.map((source, index) => `<div class="source-item">
                        <div class="source-name">${index + 1}. ${source.name}</div>
                        <div class="source-relevance">Relevance: ${(source.relevance * 100).toFixed(1)}%</div></div>`).join('')}</div>`;
                    }
                    responseArea.innerHTML = `<div class="response-container">
                    <div class="response-header"><h3>💬 Response for ${result.user_id}</h3>
                    <span class="confidence ${result.confidence}">${result.confidence}</span></div>
                    <div class="answer">${result.answer}</div>${sourcesHtml}
                    <div class="timestamp">⏰ ${timestamp}</div></div>`;
                } else {
                    responseArea.innerHTML = `<div class="error"><strong>❌ Error:</strong> ${data.error}</div>`;
                }
            } catch (error) {
                loading.style.display = 'none';
                responseArea.innerHTML = `<div class="error"><strong>❌ Network Error:</strong> Could not connect to the server.</div>`;
            }
        });
    </script>
</body>
</html>"""
    
    with open("templates/index.html", "w", encoding="utf-8") as f:
        f.write(html_template)

# FastAPI Application
app = FastAPI(title="Product Query Bot", version="1.0.0")

# Create web files on startup
create_web_files()

# Setup templates
templates = Jinja2Templates(directory="templates")

# Global instances
vector_db_manager = None
multi_agent_system = None
output_logger = None

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    global vector_db_manager, multi_agent_system, output_logger
    
    output_logger = OutputLogger()
    output_logger.log("Starting Product Query Bot with persistent vector database...")
    output_logger.log(f"Configuration: TOP_K={Config.TOP_K}, SIMILARITY_THRESHOLD={Config.SIMILARITY_THRESHOLD}")
    
    # Initialize vector database with persistence
    vector_db_manager = VectorDatabaseManager(logger=output_logger)
    
    if vector_db_manager.load_and_index_products():
        output_logger.log("Vector database loaded/created successfully!")
        
        # Log database info
        db_info = vector_db_manager.get_database_info()
        output_logger.log(f"Database info: {db_info}")
    else:
        output_logger.log("Failed to load/create vector database!", "ERROR")
        return
    
    multi_agent_system = MultiAgentSystem(vector_db_manager, output_logger)
    output_logger.log("Product Query Bot is ready!")

@app.get("/", response_class=HTMLResponse)
async def web_interface(request: Request):
    """Serve the web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/config")
async def get_configuration():
    """Get current configuration"""
    return {
        "vector_db_persist_dir": Config.VECTOR_DB_PERSIST_DIR,
        "collection_name": Config.COLLECTION_NAME,
        "embedding_model": Config.EMBEDDING_MODEL,
        "top_k": Config.TOP_K,
        "similarity_threshold": Config.SIMILARITY_THRESHOLD,
        "max_context_length": Config.MAX_CONTEXT_LENGTH,
        "max_answer_sentences": Config.MAX_ANSWER_SENTENCES,
        "min_sentence_length": Config.MIN_SENTENCE_LENGTH,
        "data_file_path": Config.DATA_FILE_PATH,
        "host": Config.HOST,
        "port": Config.PORT
    }

@app.get("/db-info")
async def get_database_info():
    """Get vector database information"""
    if vector_db_manager:
        return vector_db_manager.get_database_info()
    else:
        return {"status": "not_initialized", "error": "Vector database manager not initialized"}

@app.post("/web-query")
async def web_query(user_id: str = Form(...), query: str = Form(...)):
    """Handle web form submissions"""
    if not multi_agent_system:
        return {"success": False, "error": "System not initialized"}
    
    try:
        if user_id.startswith('anonymous_'):
            user_id = f"web_user_{user_id.split('_')[1]}"
        
        result = multi_agent_system.process_query(user_id, query)
        output_logger.log_query_interaction(result['user_id'], result['query'], result['answer'], result['sources'], result['confidence'])
        
        return {"success": True, "result": result}
    except Exception as e:
        output_logger.log(f"Error in web query: {e}", "ERROR")
        return {"success": False, "error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint with database info"""
    db_info = vector_db_manager.get_database_info() if vector_db_manager else {"status": "not_initialized"}
    
    return {
        "status": "healthy" if multi_agent_system else "initializing",
        "ready": multi_agent_system is not None,
        "database": db_info,
        "config": {
            "top_k": Config.TOP_K,
            "similarity_threshold": Config.SIMILARITY_THRESHOLD,
            "embedding_model": Config.EMBEDDING_MODEL
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)