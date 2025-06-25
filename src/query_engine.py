"""Query engine for semantic search and answer generation."""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from langdetect import detect
from llama_index.core.base.response.schema import Response

logger = logging.getLogger(__name__)


class QueryEngine:
    """Handle query processing and response generation."""
    
    def __init__(self, vector_store_manager):
        self.vector_store_manager = vector_store_manager
        self.query_history = []
    
    def detect_query_language(self, query: str) -> str:
        """Detect the language of the query."""
        try:
            return detect(query)
        except:
            return "en"
    
    def get_language_prompt(self, language: str) -> str:
        """Get system prompt based on detected language."""
        prompts = {
            "ru": """Ты помощник для ответов на вопросы на основе предоставленных документов.
            
Инструкции:
1. Отвечай на русском языке
2. Используй только информацию из предоставленных документов
3. Если информации недостаточно, так и скажи
4. Указывай источники своих ответов
5. Будь конкретным и точным""",
            
            "en": """You are an assistant for answering questions based on provided documents.
            
Instructions:
1. Answer in English
2. Use only information from the provided documents
3. If information is insufficient, say so
4. Cite your sources
5. Be specific and accurate"""
        }
        return prompts.get(language, prompts["en"])
    
    def calculate_confidence_score(self, response: Response) -> float:
        """Calculate confidence score based on response quality."""
        if not response.source_nodes:
            return 0.0
        
        # Calculate average similarity scores
        scores = [node.score for node in response.source_nodes if hasattr(node, 'score') and node.score]
        if not scores:
            return 0.5  # Default moderate confidence
        
        avg_score = sum(scores) / len(scores)
        
        # Convert to 0-100 scale
        confidence = min(100, max(0, avg_score * 100))
        
        # Adjust based on response length and source count
        if len(response.response) < 50:
            confidence *= 0.8  # Reduce confidence for very short responses
        
        if len(response.source_nodes) >= 3:
            confidence *= 1.1  # Boost confidence for multiple sources
        
        return min(100, confidence)
    
    def format_sources(self, source_nodes: List[Any]) -> List[Dict[str, Any]]:
        """Format source nodes for display."""
        sources = []
        for i, node in enumerate(source_nodes):
            source_info = {
                "index": i + 1,
                "score": getattr(node, 'score', 0.0),
                "filename": node.metadata.get("filename", "Unknown"),
                "title": node.metadata.get("title", "Unknown"),
                "author": node.metadata.get("author", "Unknown"),
                "text_snippet": node.text[:200] + "..." if len(node.text) > 200 else node.text
            }
            sources.append(source_info)
        return sources
    
    def process_query(
        self, 
        query: str, 
        similarity_top_k: int = 5,
        similarity_threshold: float = 0.7,
        metadata_filters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Process a query and return structured response."""
        
        # Detect query language
        query_language = self.detect_query_language(query)
        
        # Get query engine
        query_engine = self.vector_store_manager.get_query_engine(
            similarity_top_k=similarity_top_k,
            filters=metadata_filters
        )
        
        if not query_engine:
            return {
                "success": False,
                "error": "Query engine not available. Please upload documents first."
            }
        
        try:
            # Execute query
            response = query_engine.query(query)
            
            # Filter by similarity threshold
            filtered_sources = [
                node for node in response.source_nodes 
                if hasattr(node, 'score') and node.score >= similarity_threshold
            ]
            
            if not filtered_sources:
                return {
                    "success": True,
                    "answer": "Не найдено релевантных документов для ответа на ваш вопрос." if query_language == "ru" else "No relevant documents found to answer your question.",
                    "confidence": 0.0,
                    "sources": [],
                    "language": query_language,
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Update response with filtered sources
            response.source_nodes = filtered_sources
            
            # Calculate confidence
            confidence = self.calculate_confidence_score(response)
            
            # Format sources
            sources = self.format_sources(filtered_sources)
            
            # Prepare result
            result = {
                "success": True,
                "answer": response.response,
                "confidence": confidence,
                "sources": sources,
                "language": query_language,
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to history
            self.query_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "error": f"Error processing query: {str(e)}"
            }
    
    def get_query_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent query history."""
        return self.query_history[-limit:] if self.query_history else []
    
    def clear_history(self):
        """Clear query history."""
        self.query_history = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get query statistics."""
        if not self.query_history:
            return {
                "total_queries": 0,
                "avg_confidence": 0.0,
                "language_distribution": {},
                "recent_queries": 0
            }
        
        successful_queries = [q for q in self.query_history if q.get("success", False)]
        
        # Language distribution
        languages = [q.get("language", "unknown") for q in successful_queries]
        lang_dist = {}
        for lang in languages:
            lang_dist[lang] = lang_dist.get(lang, 0) + 1
        
        # Average confidence
        confidences = [q.get("confidence", 0) for q in successful_queries]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "total_queries": len(self.query_history),
            "successful_queries": len(successful_queries),
            "avg_confidence": round(avg_confidence, 2),
            "language_distribution": lang_dist,
            "recent_queries": len([q for q in self.query_history[-24:] if q.get("success", False)])
        }