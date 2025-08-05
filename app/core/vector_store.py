# app/core/vector_store.py
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import asyncio

from app.config.settings import settings
from app.models.response_models import DocumentChunk

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Simplified vector store for cloud deployment
    Uses basic similarity calculation instead of FAISS for better compatibility
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize vector store
        
        Args:
            dimension: Dimension of embeddings
        """
        self.dimension = dimension
        self.chunks: List[DocumentChunk] = []
        self.embeddings: List[List[float]] = []
        logger.info(f"Initialized vector store with dimension {dimension}")
    
    def add_chunks(self, chunks: List[DocumentChunk], document_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of DocumentChunk objects with embeddings
            document_metadata: Optional metadata about the document
        """
        if not chunks:
            logger.warning("No chunks provided to add to vector store")
            return
            
        # Store document metadata if provided
        if document_metadata:
            logger.info(f"Adding chunks with document metadata: {document_metadata}")
            
        for chunk in chunks:
            if chunk.embedding and len(chunk.embedding) == self.dimension:
                # Add document metadata to chunk metadata if provided
                if document_metadata:
                    chunk.metadata.update(document_metadata)
                
                self.chunks.append(chunk)
                self.embeddings.append(chunk.embedding)
            else:
                logger.warning(f"Chunk {chunk.chunk_id} has invalid embedding dimension")
        
        logger.info(f"Added {len(chunks)} chunks to vector store. Total chunks: {len(self.chunks)}")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Convert to numpy arrays for easier calculation
            a = np.array(vec1)
            b = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            similarity = dot_product / (norm_a * norm_b)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar chunks using cosine similarity
        
        Args:
            query_embedding: Query vector to search for
            top_k: Number of top results to return
            
        Returns:
            List of tuples (chunk, similarity_score) sorted by similarity
        """
        if not self.chunks or not query_embedding:
            logger.warning("No chunks in vector store or invalid query embedding")
            return []
        
        if len(query_embedding) != self.dimension:
            logger.error(f"Query embedding dimension {len(query_embedding)} doesn't match store dimension {self.dimension}")
            return []
        
        # Calculate similarities
        similarities = []
        for i, chunk_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((self.chunks[i], similarity))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Apply similarity threshold filter
        threshold = settings.SIMILARITY_THRESHOLD
        filtered_results = [
            (chunk, score) for chunk, score in similarities 
            if score >= threshold
        ]
        
        # Return top_k results
        results = filtered_results[:top_k]
        
        logger.info(f"Found {len(results)} chunks above threshold {threshold} for search query")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store
        
        Returns:
            Dictionary with store statistics
        """
        if not self.chunks:
            return {
                "status": "empty",
                "total_chunks": 0,
                "dimension": self.dimension
            }
        
        # Calculate some basic statistics
        chunk_lengths = [len(chunk.text) for chunk in self.chunks]
        avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths)
        
        return {
            "status": "populated",
            "total_chunks": len(self.chunks),
            "dimension": self.dimension,
            "avg_chunk_length": round(avg_chunk_length, 2),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "similarity_threshold": settings.SIMILARITY_THRESHOLD
        }
    
    def clear(self) -> None:
        """Clear all chunks and embeddings from the vector store"""
        self.chunks.clear()
        self.embeddings.clear()
        logger.info("Cleared vector store")
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Get a specific chunk by its ID
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            DocumentChunk if found, None otherwise
        """
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def remove_chunk(self, chunk_id: str) -> bool:
        """
        Remove a chunk from the vector store
        
        Args:
            chunk_id: ID of the chunk to remove
            
        Returns:
            True if chunk was removed, False if not found
        """
        for i, chunk in enumerate(self.chunks):
            if chunk.chunk_id == chunk_id:
                del self.chunks[i]
                del self.embeddings[i]
                logger.info(f"Removed chunk {chunk_id} from vector store")
                return True
        
        logger.warning(f"Chunk {chunk_id} not found in vector store")
        return False
    
    def __len__(self) -> int:
        """Return the number of chunks in the store"""
        return len(self.chunks)
    
    def __bool__(self) -> bool:
        """Return True if store has chunks, False otherwise"""
        return len(self.chunks) > 0
