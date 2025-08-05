# app/main.py
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from datetime import datetime
from typing import Optional

from app.config.settings import settings
from app.models.request_models import DocumentProcessRequest, QuestionRequest
from app.models.response_models import DocumentProcessResponse, DocumentProcessAdvancedResponse, SimpleAnswersResponse, QuestionResponse
from app.core.document_processor import DocumentProcessor
from app.core.intelligent_processor import IntelligentProcessor
from app.utils.exceptions import DocumentProcessingError
from app.utils.auth import require_auth

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Intelligent Document Query System",
    description="Advanced document processing system with AI-powered question answering using embeddings and LLM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
document_processor = DocumentProcessor()
intelligent_processor = IntelligentProcessor()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Intelligent Document Query System",
        "status": "running",
        "capabilities": [
            "Document processing (PDF, DOCX)",
            "Text embedding generation",
            "Vector similarity search",
            "AI-powered question answering",
            "Advanced document queries"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "document-processor",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/hackrx/run")
async def process_document_with_qa(
    request: DocumentProcessRequest, 
    authorization: Optional[str] = Header(None)
):
    """
    Main endpoint: Process document with AI-powered question answering
    
    Returns only the answers array in the format: {"answers": [...]}
    
    This endpoint:
    1. Downloads and processes document
    2. Generates embeddings for text chunks
    3. Stores chunks in vector database
    4. Answers questions using retrieval-augmented generation (RAG)
    """
    # Require authentication
    require_auth(authorization)
    
    request_id = f"req_{int(datetime.now().timestamp() * 1000)}"
    
    logger.info(f"[{request_id}] Processing document: {request.documents}")
    logger.info(f"[{request_id}] Questions to answer: {len(request.questions)}")
    
    try:
        # Process document with embeddings and Q&A
        result = await intelligent_processor.process_document_with_qa(
            document_url=request.documents,
            questions=request.questions,
            request_id=request_id
        )
        
        logger.info(f"[{request_id}] Processing completed successfully. "
                   f"Chunks: {len(result['chunks'])}, "
                   f"Questions answered: {len(result['answers'])}")
        
        # Return only the answers array as requested
        answers = [qa.answer for qa in result['answers']]
        return {"answers": answers}
        
    except DocumentProcessingError as e:
        logger.error(f"[{request_id}] Document processing error: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Failed to process document: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while processing the document"
        )


@app.post("/hackrx/answers", response_model=SimpleAnswersResponse)
async def get_simple_answers(
    request: DocumentProcessRequest,
    authorization: Optional[str] = Header(None)
) -> SimpleAnswersResponse:
    # Require authentication
    require_auth(authorization)
    """
    Simplified endpoint that returns only answers in the required format
    
    This endpoint processes documents and returns only the answers array.
    Supports Authorization header as required.
    """
    request_id = f"req_{int(datetime.now().timestamp() * 1000)}"
    
    logger.info(f"[{request_id}] Processing document for simple answers: {request.documents}")
    logger.info(f"[{request_id}] Questions to answer: {len(request.questions)}")
    
    if authorization:
        logger.info(f"[{request_id}] Authorization header provided")
    
    try:
        # Process document with embeddings and Q&A
        result = await intelligent_processor.process_document_with_qa(
            document_url=request.documents,
            questions=request.questions,
            request_id=request_id
        )
        
        # Extract just the answer strings
        answers = [qa.answer for qa in result['answers']]
        
        logger.info(f"[{request_id}] Processing completed. Generated {len(answers)} answers")
        
        return SimpleAnswersResponse(answers=answers)
        
    except DocumentProcessingError as e:
        logger.error(f"[{request_id}] Document processing error: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Failed to process document: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while processing the document"
        )


# Additional utility endpoints for debugging and monitoring


@app.post("/hackrx/legacy", response_model=DocumentProcessResponse)
async def process_document_legacy(request: DocumentProcessRequest) -> DocumentProcessResponse:
    """
    Legacy endpoint: Process document and extract text chunks only (for backward compatibility)
    
    This endpoint:
    1. Downloads document from provided URL
    2. Extracts text content from PDF/DOCX
    3. Splits text into manageable chunks
    4. Returns processed chunks and metadata (no AI answering)
    """
    request_id = f"req_{int(datetime.now().timestamp() * 1000)}"
    
    logger.info(f"[{request_id}] Processing document: {request.documents}")
    
    try:
        # Process the document
        result = await document_processor.process_document_from_url(
            document_url=request.documents,
            request_id=request_id
        )
        
        logger.info(f"[{request_id}] Document processed successfully. "
                   f"Chunks: {len(result['chunks'])}, "
                   f"Total characters: {result['metadata']['total_characters']}")
        
        # For Phase 1, we'll just return the chunks and some basic info
        # In later phases, these chunks will be used for embedding and querying
        return DocumentProcessResponse(
            success=True,
            message="Document processed successfully",
            chunks=result['chunks'],
            metadata={
                "request_id": request_id,
                "document_url": request.documents,
                "total_chunks": len(result['chunks']),
                "total_characters": result['metadata']['total_characters'],
                "file_type": result['metadata']['file_type'],
                "processing_time_seconds": result['metadata']['processing_time'],
                "timestamp": datetime.now().isoformat()
            },
            questions_placeholder=request.questions  # Just store for now
        )
        
    except DocumentProcessingError as e:
        logger.error(f"[{request_id}] Document processing error: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Failed to process document: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while processing the document"
        )

@app.post("/hackrx/validate")
async def validate_document_url(request: DocumentProcessRequest):
    """
    Validate if a document URL is accessible and supported
    """
    try:
        is_valid = await document_processor.validate_document_url(request.documents)
        
        return {
            "valid": is_valid,
            "url": request.documents,
            "message": "Document is accessible" if is_valid else "Document is not accessible",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Document validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to validate document: {str(e)}"
        )


@app.post("/hackrx/search")
async def search_current_document(query: str, top_k: int = 5):
    """
    Search the currently loaded document using semantic similarity
    
    Args:
        query: Search query
        top_k: Number of results to return
    """
    try:
        if not intelligent_processor.current_vector_store:
            raise HTTPException(
                status_code=400,
                detail="No document loaded. Please process a document first using /hackrx/run"
            )
        
        results = await intelligent_processor.search_document(query, top_k)
        
        return {
            "query": query,
            "results": [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "similarity": float(score),
                    "page_number": chunk.page_number,
                    "metadata": chunk.metadata
                }
                for chunk, score in results
            ],
            "total_results": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Search failed: {str(e)}"
        )


@app.get("/hackrx/vector-store/stats")
async def get_vector_store_stats():
    """Get statistics about the current vector store"""
    try:
        stats = intelligent_processor.get_vector_store_stats()
        return {
            "vector_store_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


@app.post("/hackrx/question", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    authorization: Optional[str] = Header(None)
) -> QuestionResponse:
    # Require authentication
    require_auth(authorization)
    """
    Ask a single question about the processed document
    
    This endpoint allows you to ask individual questions about documents
    that have already been processed and stored in the vector database.
    """
    try:
        # Check if we have any documents loaded
        if not intelligent_processor.current_vector_store or not intelligent_processor.current_vector_store.chunks:
            return QuestionResponse(
                status="error",
                answer="No document has been processed yet. Please process a document first using the /hackrx/run endpoint."
            )
        
        # Process the question
        response = intelligent_processor.process_question(request.question)
        
        return QuestionResponse(
            status="success",
            answer=response.answer
        )
        
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return QuestionResponse(
            status="error",
            answer=f"Sorry, I encountered an error while processing your question: {str(e)}"
        )


@app.get("/hackrx/debug/chunks")
async def debug_chunks():
    """Debug endpoint to check what chunks are stored"""
    try:
        if not intelligent_processor.current_vector_store:
            return {"error": "No vector store loaded"}
        
        chunks = intelligent_processor.current_vector_store.chunks[:5]  # First 5 chunks
        return {
            "total_chunks": len(intelligent_processor.current_vector_store.chunks),
            "sample_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    "has_embedding": chunk.embedding is not None,
                    "embedding_length": len(chunk.embedding) if chunk.embedding else 0
                }
                for chunk in chunks
            ]
        }
    except Exception as e:
        logger.error(f"Debug chunks error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import os
    
    # Use PORT environment variable for deployment platforms like Render
    port = int(os.getenv("PORT", settings.API_PORT))
    host = os.getenv("HOST", settings.API_HOST)
    
    # For production deployment, ensure host is 0.0.0.0
    if os.getenv("RENDER") or os.getenv("PORT"):  # Render sets PORT environment variable
        host = "0.0.0.0"
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=settings.DEBUG,
        log_level="info"
    )
