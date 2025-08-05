# app/models/response_models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class QuestionResponse(BaseModel):
    """Response model for single question answers"""
    
    status: str = Field(..., description="Status of the response")
    answer: str = Field(..., description="The detailed answer to the question")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "answer": "According to the policy document, the grace period for submitting insurance claims is 30 days from the date of the incident. This grace period allows claimants sufficient time to gather necessary documentation and submit their claims properly."
            }
        }

class DocumentChunk(BaseModel):
    """Model for document text chunks"""
    
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="Text content of the chunk")
    start_char: int = Field(..., description="Starting character position in original document")
    end_char: int = Field(..., description="Ending character position in original document")
    page_number: Optional[int] = Field(None, description="Page number if available")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the chunk")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the chunk text")
    
    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "chunk_0001",
                "text": "This is a sample text chunk from the document...",
                "start_char": 0,
                "end_char": 500,
                "page_number": 1,
                "metadata": {
                    "chunk_index": 0,
                    "word_count": 75,
                    "char_count": 500
                },
                "embedding": [0.1, 0.2, -0.3]  # Sample embedding values
            }
        }

class DocumentProcessResponse(BaseModel):
    """Response model for document processing"""
    
    success: bool = Field(..., description="Whether the processing was successful")
    message: str = Field(..., description="Status message")
    chunks: List[DocumentChunk] = Field(default=[], description="List of text chunks extracted from the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
    questions_placeholder: List[str] = Field(default=[], description="Placeholder for questions (legacy compatibility)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Document processed successfully",
                "chunks": [
                    {
                        "chunk_id": "chunk_0001",
                        "text": "Sample text content...",
                        "start_char": 0,
                        "end_char": 500,
                        "page_number": 1,
                        "metadata": {"word_count": 75}
                    }
                ],
                "metadata": {
                    "request_id": "req_1234567890",
                    "total_chunks": 15,
                    "total_characters": 7500,
                    "file_type": "pdf",
                    "processing_time_seconds": 2.5
                },
                "questions_placeholder": []
            }
        }


# Advanced models for Q&A functionality

class QuestionAnswer(BaseModel):
    """Model for a single question-answer pair"""
    
    question: str = Field(..., description="The question asked")
    answer: str = Field(..., description="Generated answer based on document context")
    confidence: float = Field(..., description="Confidence score (0-1)")
    context_chunks_used: int = Field(..., description="Number of context chunks used")
    sources: List[Dict[str, Any]] = Field(default=[], description="Source chunks with metadata")
    method: str = Field(..., description="Method used for answer generation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the main topic of this document?",
                "answer": "The document discusses data privacy policies...",
                "confidence": 0.85,
                "context_chunks_used": 3,
                "sources": [
                    {
                        "chunk_id": "chunk_0001",
                        "similarity": 0.92,
                        "page_number": 1
                    }
                ],
                "method": "openai"
            }
        }


class SimpleAnswersResponse(BaseModel):
    """Simple response model that only returns answers array"""
    
    answers: List[str] = Field(..., description="List of answers to the questions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answers": [
                    "A grace period of thirty days is provided for premium payment after the due date.",
                    "There is a waiting period of thirty-six (36) months for pre-existing diseases.",
                    "Yes, the policy covers maternity expenses with 24 months continuous coverage requirement."
                ]
            }
        }


class DocumentProcessAdvancedResponse(BaseModel):
    """Enhanced response model with embeddings and Q&A capabilities"""
    
    success: bool = Field(..., description="Whether the processing was successful")
    message: str = Field(..., description="Status message")
    chunks: List[DocumentChunk] = Field(default=[], description="List of text chunks with embeddings")
    answers: List[QuestionAnswer] = Field(default=[], description="Answers to provided questions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata including embeddings info")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Document processed successfully with AI-powered Q&A",
                "chunks": [
                    {
                        "chunk_id": "chunk_0001",
                        "text": "Sample text content...",
                        "start_char": 0,
                        "end_char": 500,
                        "page_number": 1,
                        "metadata": {"word_count": 75},
                        "embedding": [0.1, 0.2, -0.3]
                    }
                ],
                "answers": [
                    {
                        "question": "What is the main topic?",
                        "answer": "The main topic is...",
                        "confidence": 0.85,
                        "context_chunks_used": 3,
                        "sources": [],
                        "method": "openai"
                    }
                ],
                "metadata": {
                    "request_id": "req_1234567890",
                    "total_chunks": 15,
                    "total_characters": 7500,
                    "file_type": "pdf",
                    "processing_time_seconds": 2.5,
                    "embedding_model": "all-MiniLM-L6-v2",
                    "vector_store_populated": True,
                    "questions_answered": 2
                }
            }
        }


class SimpleAnswersResponse(BaseModel):
    """Simplified response model that matches the required output format"""
    
    answers: List[str] = Field(..., description="List of answers to the questions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answers": [
                    "A grace period of thirty days is provided for premium payment after the due date...",
                    "There is a waiting period of thirty-six (36) months of continuous coverage..."
                ]
            }
        }
