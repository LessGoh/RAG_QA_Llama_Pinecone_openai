"""Vector store integration with Pinecone."""

import logging
from typing import List, Optional, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.vector_stores.types import VectorStoreQuery, MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage Pinecone vector store operations."""
    
    def __init__(self, api_key: str, environment: str, index_name: str):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.pc = None
        self.pinecone_index = None
        self.vector_store = None
        self.index = None
        
        # Initialize OpenAI models
        self.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        self.llm = OpenAI(model="gpt-4", temperature=0.2)
        
        # Set global settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
    
    def initialize_pinecone(self) -> bool:
        """Initialize Pinecone connection."""
        try:
            self.pc = Pinecone(api_key=self.api_key)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            return False
    
    def create_index(self, dimension: int = 1536, metric: str = "cosine") -> bool:
        """Create a new Pinecone index if it doesn't exist."""
        try:
            if not self.pc:
                if not self.initialize_pinecone():
                    return False
            
            existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Created new index: {self.index_name}")
            else:
                logger.info(f"Index {self.index_name} already exists")
            
            return True
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    def connect_to_index(self) -> bool:
        """Connect to existing Pinecone index."""
        try:
            if not self.pc:
                if not self.initialize_pinecone():
                    return False
            
            # Get index host
            index_description = self.pc.describe_index(self.index_name)
            index_host = index_description.host
            
            # Connect to index
            self.pinecone_index = self.pc.Index(host=index_host)
            
            # Create vector store
            self.vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
            
            logger.info(f"Connected to index: {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to index: {e}")
            return False
    
    def create_vector_index(self, documents: List[Any]) -> bool:
        """Create VectorStoreIndex from documents."""
        try:
            if not self.vector_store:
                if not self.connect_to_index():
                    return False
            
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context
            )
            
            logger.info(f"Created vector index with {len(documents)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
            return False
    
    def load_existing_index(self) -> bool:
        """Load existing VectorStoreIndex."""
        try:
            if not self.vector_store:
                if not self.connect_to_index():
                    return False
            
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            self.index = VectorStoreIndex(
                nodes=[],
                storage_context=storage_context
            )
            
            logger.info("Loaded existing vector index")
            return True
        except Exception as e:
            logger.error(f"Failed to load existing index: {e}")
            return False
    
    def get_query_engine(self, similarity_top_k: int = 5, filters: Optional[Dict] = None):
        """Get query engine for searching."""
        if not self.index:
            return None
        
        # Преобразование filters (dict) в MetadataFilters, если нужно
        metadata_filters = None
        if filters and isinstance(filters, dict) and len(filters) > 0:
            filter_list = []
            for key, value in filters.items():
                # По умолчанию оператор EQ (равенство)
                filter_list.append(MetadataFilter(key=key, value=value, operator=FilterOperator.EQ))
            metadata_filters = MetadataFilters(
                filters=filter_list,
                condition=FilterCondition.AND
            )
        
        return self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            filters=metadata_filters
        )
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self.pinecone_index:
            return {}
        
        try:
            stats = self.pinecone_index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}
    
    def delete_index(self) -> bool:
        """Delete the Pinecone index."""
        try:
            if not self.pc:
                if not self.initialize_pinecone():
                    return False
            
            self.pc.delete_index(self.index_name)
            logger.info(f"Deleted index: {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete index: {e}")
            return False