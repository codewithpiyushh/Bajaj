# app/core/intelligent_processor.py
import logging
import time
from typing import Dict, List, Any, Tuple
import asyncio

from app.core.document_processor import DocumentProcessor
from app.core.embedding_service import EmbeddingService
from app.core.vector_store import VectorStore
from app.core.llm_service import LLMService
from app.models.response_models import DocumentChunk, QuestionAnswer
from app.config.settings import settings
from app.utils.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)

class IntelligentProcessor:
    """
    Intelligent document processor that combines document processing with embeddings and Q&A
    """
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        # Vector store will be created per session/document
        self.current_vector_store: VectorStore = None
    
    async def process_document_with_qa(
        self, 
        document_url: str, 
        questions: List[str], 
        request_id: str
    ) -> Dict[str, Any]:
        """
        Main processing method: document processing + embeddings + Q&A
        
        Args:
            document_url: URL of the document to process
            questions: List of questions to answer
            request_id: Unique request identifier
            
        Returns:
            Dictionary containing chunks with embeddings and Q&A results
        """
        start_time = time.time()
        
        try:
            logger.info(f"[{request_id}] Starting intelligent document processing for: {document_url}")
            
            # Step 1: Process document
            logger.info(f"[{request_id}] Step 1: Processing document...")
            document_result = await self.document_processor.process_document_from_url(
                document_url=document_url,
                request_id=request_id
            )
            
            chunks = document_result['chunks']
            base_metadata = document_result['metadata']
            
            if not chunks:
                raise DocumentProcessingError("No text chunks extracted from document")
            
            # Step 2: Generate embeddings for chunks
            logger.info(f"[{request_id}] Step 2: Generating embeddings for {len(chunks)} chunks...")
            chunks_with_embeddings = await self.embedding_service.embed_chunks(chunks)
            
            # Step 3: Create and populate vector store
            logger.info(f"[{request_id}] Step 3: Creating vector store...")
            embedding_dim = self.embedding_service.get_embedding_dimension()
            self.current_vector_store = VectorStore(dimension=embedding_dim)
            
            self.current_vector_store.add_chunks(
                chunks_with_embeddings, 
                document_metadata=base_metadata
            )
            
            # Step 4: Answer questions using RAG (Retrieval-Augmented Generation)
            answers = []
            if questions:
                logger.info(f"[{request_id}] Step 4: Answering {len(questions)} questions...")
                answers = await self._answer_questions(questions, request_id)
            
            processing_time = time.time() - start_time
            
            # Prepare enhanced metadata
            enhanced_metadata = {
                **base_metadata,
                "embedding_model": settings.EMBEDDING_MODEL,
                "embedding_dimension": embedding_dim,
                "vector_store_populated": True,
                "questions_answered": len(answers),
                "total_processing_time_seconds": round(processing_time, 2),
                "system_type": "intelligent_document_processor"
            }
            
            logger.info(f"[{request_id}] Intelligent processing completed successfully in {processing_time:.2f}s")
            
            return {
                "chunks": chunks_with_embeddings,
                "answers": answers,
                "metadata": enhanced_metadata,
                "vector_store": self.current_vector_store  # For potential reuse
            }
            
        except Exception as e:
            logger.error(f"[{request_id}] Intelligent processing failed: {e}")
            raise DocumentProcessingError(f"Document processing failed: {e}")
    
    async def _answer_questions(self, questions: List[str], request_id: str) -> List[QuestionAnswer]:
        """
        Answer a list of questions using the populated vector store
        
        Args:
            questions: List of questions to answer
            request_id: Request identifier for logging
            
        Returns:
            List of QuestionAnswer objects
        """
        if not self.current_vector_store or not questions:
            return []
        
        answers = []
        
        for i, question in enumerate(questions, 1):
            try:
                logger.info(f"[{request_id}] Answering question {i}/{len(questions)}: {question[:50]}...")
                
                # Generate embedding for the question
                question_embedding = await self.embedding_service.generate_single_embedding(question)
                logger.info(f"[{request_id}] Generated question embedding with dimension: {len(question_embedding)}")
                
                # Retrieve relevant chunks
                retrieved_chunks = self.current_vector_store.search(
                    query_embedding=question_embedding,
                    top_k=settings.TOP_K_CHUNKS
                )
                
                logger.info(f"[{request_id}] Retrieved {len(retrieved_chunks)} chunks for question: {question[:50]}")
                if retrieved_chunks:
                    logger.info(f"[{request_id}] Top similarity scores: {[score for _, score in retrieved_chunks[:3]]}")
                
                if not retrieved_chunks:
                    # No relevant context found
                    answer = QuestionAnswer(
                        question=question,
                        answer="I couldn't find relevant information in the document to answer this question.",
                        confidence=0.0,
                        context_chunks_used=0,
                        sources=[],
                        method="no_context"
                    )
                else:
                    # Generate answer using LLM
                    llm_result = await self.llm_service.answer_question(question, retrieved_chunks)
                    
                    # Calculate confidence based on similarity scores and method
                    avg_similarity = sum(score for _, score in retrieved_chunks) / len(retrieved_chunks)
                    confidence = min(avg_similarity * 1.1, 1.0)  # Boost slightly but cap at 1.0
                    
                    if llm_result["method"] == "fallback":
                        confidence *= 0.6  # Reduce confidence for fallback method
                    
                    answer = QuestionAnswer(
                        question=question,
                        answer=llm_result["answer"],
                        confidence=round(confidence, 2),
                        context_chunks_used=llm_result["context_chunks_used"],
                        sources=llm_result["chunk_sources"],
                        method=llm_result["method"]
                    )
                
                answers.append(answer)
                logger.info(f"[{request_id}] Question {i} answered with confidence {answer.confidence}")
                
            except Exception as e:
                logger.error(f"[{request_id}] Error answering question {i}: {e}")
                # Add error answer
                error_answer = QuestionAnswer(
                    question=question,
                    answer=f"Error occurred while processing this question: {str(e)}",
                    confidence=0.0,
                    context_chunks_used=0,
                    sources=[],
                    method="error"
                )
                answers.append(error_answer)
        
        return answers
    
    async def search_document(self, query: str, top_k: int = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Search the current document using a query
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if not self.current_vector_store:
            raise ValueError("No document loaded in vector store")
        
        # Generate embedding for query
        query_embedding = await self.embedding_service.generate_single_embedding(query)
        
        # Search vector store
        results = self.current_vector_store.search(query_embedding, top_k or settings.TOP_K_CHUNKS)
        
        return results
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the current vector store"""
        if not self.current_vector_store:
            return {"status": "no_vector_store"}
        
        return self.current_vector_store.get_stats()
    
    def clear_vector_store(self):
        """Clear the current vector store"""
        if self.current_vector_store:
            self.current_vector_store.clear()
            self.current_vector_store = None
            logger.info("Vector store cleared")
    
    async def cleanup(self):
        """Clean up all resources"""
        if self.document_processor:
            await self.document_processor.close()
        
        if self.embedding_service:
            self.embedding_service.cleanup()
        
        self.clear_vector_store()
        
        logger.info("Intelligent processor cleanup completed")
