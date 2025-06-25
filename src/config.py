"""Configuration module for RAG QA System."""

import os
from typing import Optional
from pydantic import BaseModel, field_validator


class Settings(BaseModel):
    """Application settings."""
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4"
    embedding_model: str = "text-embedding-ada-002"
    
    # Pinecone Configuration
    pinecone_api_key: str
    pinecone_environment: str = "us-east1-gcp"
    pinecone_index_name: str = "qa-documents"
    
    # Application Settings
    max_file_size_mb: int = 50
    max_files_count: int = 100
    chunk_size: int = 1024
    chunk_overlap: int = 20
    similarity_top_k: int = 5
    similarity_threshold: float = 0.7
    
    # Database Settings
    database_url: str = "sqlite:///qa_system.db"
    
    def __init__(self, **kwargs):
        # Load from environment variables
        env_vars = {
            'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
            'openai_model': os.getenv('OPENAI_MODEL', 'gpt-4'),
            'embedding_model': os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002'),
            'pinecone_api_key': os.getenv('PINECONE_API_KEY', ''),
            'pinecone_environment': os.getenv('PINECONE_ENVIRONMENT', 'us-east1-gcp'),
            'pinecone_index_name': os.getenv('PINECONE_INDEX_NAME', 'qa-documents'),
            'max_file_size_mb': int(os.getenv('MAX_FILE_SIZE_MB', '50')),
            'max_files_count': int(os.getenv('MAX_FILES_COUNT', '100')),
            'chunk_size': int(os.getenv('CHUNK_SIZE', '1024')),
            'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', '20')),
            'similarity_top_k': int(os.getenv('SIMILARITY_TOP_K', '5')),
            'similarity_threshold': float(os.getenv('SIMILARITY_THRESHOLD', '0.7')),
            'database_url': os.getenv('DATABASE_URL', 'sqlite:///qa_system.db')
        }
        env_vars.update(kwargs)
        super().__init__(**env_vars)
        
        # Validate required keys
        if not self.openai_api_key or self.openai_api_key == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY must be set")
        if not self.pinecone_api_key or self.pinecone_api_key == "your_pinecone_api_key_here":
            raise ValueError("PINECONE_API_KEY must be set")


def get_settings() -> Settings:
    """Get application settings."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return Settings()