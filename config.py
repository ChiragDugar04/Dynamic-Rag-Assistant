# config.py
import os

class Config:
    # LLM & Embedding Models 
    LLM_MODEL = "qwen2.5:3b"
    EMBEDDING_MODEL = "nomic-embed-text" 

    #  Vector DB 
    CHROMA_PATH = "chroma_db_storage"
    COLLECTION_NAME = "session_docs"

    #  Ingestion: Chunking 
    PARENT_CHUNK_SIZE = 1000   
    PARENT_CHUNK_OVERLAP = 100
    CHILD_CHUNK_SIZE = 300     
    CHILD_CHUNK_OVERLAP = 30

    # Ingestion: Parallelism 
    EMBEDDING_BATCH_SIZE = 64  
    MAX_WORKERS = 4            

    #  Retrieval 
    RETRIEVAL_K = 6           
    RERANK_TOP_N = 2           
    MAX_CONTEXT_CHARS = 3000   

    #  LLM Runtime 
    LLM_TEMPERATURE = 0.1
    LLM_NUM_CTX = 1536           
    LLM_NUM_THREAD = 12         
    LLM_BASE_URL = "http://127.0.0.1:11434"

    OLLAMA_KEEP_ALIVE = "10m"
