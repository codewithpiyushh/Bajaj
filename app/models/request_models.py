# app/models/request_models.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class QuestionRequest(BaseModel):
    """Request model for asking a single question about processed documents"""
    
    question: str = Field(
        ..., 
        description="The question to ask about the processed document",
        example="What is the grace period for submitting insurance claims?"
    )
    
    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the grace period for submitting insurance claims?"
            }
        }

class DocumentProcessRequest(BaseModel):
    """Request model for document processing"""
    
    documents: str = Field(
        ..., 
        description="URL of the document to process (PDF or DOCX)",
        example="https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    )
    
    @validator('documents')
    def validate_documents_url(cls, v):
        if not v or not v.strip():
            raise ValueError('Document URL cannot be empty')
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Document URL must start with http:// or https://')
        return v.strip()
    
    questions: List[str] = Field(
        default=[],
        description="List of questions to be answered (for future phases)",
        example=["What is the main topic?", "What are the key findings?"]
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
                "questions": [
                    "What is the main topic of this document?",
                    "What are the key findings?",
                    "Who are the authors?"
                ]
            }
        }
