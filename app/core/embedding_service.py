# app/core/embedding_service.py
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import asyncio
import hashlib

from app.config.settings import settings
from app.models.response_models import DocumentChunk

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Simplified embedding service for cloud deployment
    Uses basic text hashing as fallback when advanced embeddings aren't available
    """
    
    def __init__(self):
        self.embedding_dimension = 384  # Standard dimension
        logger.info("Initialized simplified embedding service")
        
    def _create_simple_embedding(self, text: str) -> List[float]:
        """Create a simple hash-based embedding for text"""
        # Simple text-based embedding using character frequencies and hashing
        text = text.lower().strip()
        
        # Create a basic feature vector
        features = []
        
        # Text length feature
        features.append(len(text) / 1000.0)
        
        # Character frequency features
        char_counts = {}
        for char in 'abcdefghijklmnopqrstuvwxyz0123456789 .':
            char_counts[char] = text.count(char) / max(len(text), 1)
            features.append(char_counts[char])
        
        # Word count features
        words = text.split()
        features.append(len(words) / 100.0)
        
        # Hash-based features for uniqueness
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert hash bytes to float features
        for i in range(0, len(hash_bytes), 2):
            if len(features) < self.embedding_dimension:
                byte_val = hash_bytes[i] if i < len(hash_bytes) else 0
                features.append(byte_val / 255.0)
        
        # Pad or truncate to exact dimension
        while len(features) < self.embedding_dimension:
            features.append(0.0)
        
        return features[:self.embedding_dimension]
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self._create_simple_embedding(text)
    
    async def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text (alias for embed_text)"""
        return await self.embed_text(text)
    
    async def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for document chunks"""
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        for chunk in chunks:
            chunk.embedding = self._create_simple_embedding(chunk.text)
        
        logger.info(f"Generated {self.embedding_dimension}-dimensional embeddings")
        return chunks
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.embedding_dimension
