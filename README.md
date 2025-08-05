# Intelligent Document Query System

An advanced AI-powered document processing system that extracts insights from PDF and DOCX files and answers questions using state-of-the-art embeddings and Large Language Models (LLM).

## � System Flow Overview

### Complete Processing Pipeline

```mermaid
graph TD
    A[POST Request with Document URL + Questions] --> B[FastAPI Router]
    B --> C[Intelligent Processor]
    
    C --> D[Step 1: Document Processing]
    D --> D1[Download Document from URL]
    D1 --> D2[Extract Text using Multiple Methods]
    D2 --> D3[Split into Chunks ~500 words]
    D3 --> D4[Create DocumentChunk Objects]
    
    D4 --> E[Step 2: Embedding Generation]
    E --> E1[Load sentence-transformers Model]
    E1 --> E2[Convert Each Chunk to Vector]
    E2 --> E3[384-dimensional Embeddings]
    
    E3 --> F[Step 3: Vector Store Setup]
    F --> F1[Initialize FAISS Index]
    F1 --> F2[Add Chunk Embeddings to Index]
    F2 --> F3[Store Metadata Mapping]
    
    F3 --> G[Step 4: Question Processing]
    G --> G1[For Each Question]
    G1 --> G2[Convert Question to Embedding]
    G2 --> G3[Search Similar Chunks FAISS]
    G3 --> G4[Retrieve Top-K Relevant Chunks]
    
    G4 --> H[Step 5: Answer Generation]
    H --> H1[Combine Retrieved Chunks as Context]
    H1 --> H2[Create Optimized Prompt]
    H2 --> H3[Send to Gemini 2.5 Flash]
    H3 --> H4[Generate Direct Answer]
    H4 --> H5[Clean & Format Response]
    
    H5 --> I[Step 6: Response Assembly]
    I --> I1[Collect All Answers]
    I1 --> I2[Add Metadata Chunks, Sources]
    I2 --> I3[Format JSON Response]
    
    I3 --> J[Return to Client]
    
    style A fill:#e1f5fe
    style J fill:#c8e6c9
    style C fill:#fff3e0
    style H3 fill:#f3e5f5
```

### What Happens When You Send a POST Request?

#### 🔍 **Detailed Step-by-Step Process**

1. **Request Reception** (`/hackrx/answers` or `/hackrx/run`)
   - FastAPI receives your POST request with document URL and questions
   - Request validation ensures proper format
   - Unique request ID generated for tracking

2. **Document Processing** (document_processor.py)
   - **Download**: Document fetched from provided URL
   - **Text Extraction**: Multiple methods used (PyMuPDF, pdfplumber, PyPDF2)
   - **Chunking**: Text split into ~500-word segments with overlap
   - **Object Creation**: Each chunk becomes a DocumentChunk with metadata

3. **Embedding Generation** (embedding_service.py)
   - **Model Loading**: sentence-transformers/all-MiniLM-L6-v2 loaded
   - **Vectorization**: Each text chunk converted to 384-dimensional vector
   - **Semantic Encoding**: Captures meaning and context of text

4. **Vector Store Creation** (vector_store.py)
   - **FAISS Index**: High-performance similarity search index created
   - **Vector Storage**: All chunk embeddings added to searchable index
   - **Metadata Mapping**: Links between vectors and original chunks maintained

5. **Question Processing Loop** (For each question)
   - **Question Embedding**: Question converted to same vector space
   - **Similarity Search**: FAISS finds most relevant chunks (cosine similarity)
   - **Context Retrieval**: Top-K chunks (default: 5) selected as context

6. **Answer Generation** (llm_service.py)
   - **Context Assembly**: Retrieved chunks combined into coherent context
   - **Prompt Engineering**: Optimized prompt created for Gemini
   - **LLM Call**: Gemini 2.5 Flash generates direct, factual answer
   - **Response Cleaning**: Answer formatted without "context" references

7. **Response Assembly**
   - **Answer Collection**: All answers compiled
   - **Metadata Addition**: Source chunks, similarity scores, page numbers added
   - **JSON Formatting**: Response structured per API specification

## �🚀 Features

- **Advanced Document Processing**: Extract text from PDF and DOCX files with multiple extraction methods
- **AI-Powered Embeddings**: Convert text chunks into vector embeddings for semantic understanding
- **Intelligent Question Answering**: Use Gemini 2.5 Flash to answer complex questions about your documents
- **Vector Similarity Search**: Find relevant information using semantic search capabilities
- **RESTful API**: Clean, well-documented API endpoints for easy integration
- **Robust Error Handling**: Comprehensive error handling with detailed logging
- **Scalable Architecture**: Modular design supporting different LLMs and vector stores

## � Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Gemini API key (get one from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd "c:\Users\Nikita\Desktop\Hackathon\Bajaj"
   ```

2. **Install dependencies**
   ```bash
   pip install -r app\requirements.txt
   ```

3. **Configure API Key**
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   LLM_MODEL=gemini-2.0-flash-exp
   ```

4. **Start the application**
   ```bash
   python app\run.py
   ```

The API will be available at `http://127.0.0.1:8000`

## 🔧 Technical Deep Dive

### Architecture Components

#### 1. **FastAPI Application** (`app/main.py`)
- **Purpose**: HTTP server and request routing
- **Key Features**: CORS support, automatic documentation, request validation
- **Endpoints**: `/hackrx/answers`, `/hackrx/run`, `/hackrx/search`

#### 2. **Document Processor** (`app/core/document_processor.py`)
- **PDF Processing**: PyMuPDF (primary), pdfplumber (fallback), PyPDF2 (backup)
- **DOCX Processing**: python-docx for Word documents
- **Text Chunking**: Smart segmentation with ~500 words per chunk, 50-word overlap
- **Metadata Extraction**: Page numbers, document type, file size

#### 3. **Embedding Service** (`app/core/embedding_service.py`)
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Dimension**: 384 (optimized for semantic similarity)
- **Processing**: Batch processing for efficiency
- **Caching**: In-memory model caching for performance

#### 4. **Vector Store** (`app/core/vector_store.py`)
- **Backend**: FAISS (Facebook AI Similarity Search)
- **Index Type**: IndexFlatIP (Inner Product for cosine similarity)
- **Search**: Fast approximate nearest neighbor search
- **Threshold**: Configurable similarity threshold (default: 0.3)

#### 5. **LLM Service** (`app/core/llm_service.py`)
- **Primary LLM**: Google Gemini 2.5 Flash
- **Fallback**: Keyword-based extraction
- **Prompt Engineering**: Optimized for direct, factual answers
- **Temperature**: Low (0.3) for consistent, focused responses

#### 6. **Intelligent Processor** (`app/core/intelligent_processor.py`)
- **Orchestration**: Coordinates all components in RAG pipeline
- **Session Management**: Per-request vector store creation
- **Error Handling**: Graceful degradation with fallbacks
- **Performance**: Async processing for concurrent operations

### Data Flow Technical Details

#### **Document Processing Pipeline**
```
URL → Download → Extract Text → Clean → Chunk → Validate → Store
```

#### **Embedding Pipeline**
```
Text Chunks → Tokenize → Model Inference → 384D Vectors → Normalize → Index
```

#### **RAG (Retrieval-Augmented Generation) Pipeline**
```
Question → Embed → Search → Retrieve → Context → LLM → Answer → Format
```

### Performance Characteristics

- **Document Processing**: ~2-5 seconds for typical PDFs (10-50 pages)
- **Embedding Generation**: ~0.1 seconds per chunk (batched)
- **Vector Search**: <50ms for similarity search
- **LLM Response**: ~1-3 seconds per question
- **Memory Usage**: ~100MB base + ~1MB per 1000 chunks

### Error Handling Strategy

1. **Document Extraction**: Multiple methods tried in sequence
2. **LLM Failures**: Automatic fallback to keyword extraction
3. **Network Issues**: Timeout handling and retry logic
4. **Invalid Input**: Comprehensive validation with clear error messages

## 🌐 API Endpoints

### Main Endpoints

#### 1. Simple Question Answering (Recommended)
**POST** `/hackrx/answers`

**Input:**
```json
{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the main topic of this document?",
        "What are the key findings?"
    ]
}
```

**Output:**
```json
{
    "answers": [
        "The document discusses...",
        "The key findings include..."
    ]
}
```

#### 2. Advanced Processing with Metadata
**POST** `/hackrx/run`

Returns detailed information including document chunks, embeddings, confidence scores, and source attribution.

#### 3. Document Search
**POST** `/hackrx/search`

Search within a processed document using natural language queries.

**Parameters:**
- `query`: Search query string
- `top_k`: Number of results to return (default: 5)

#### 4. Legacy Text Extraction
**POST** `/hackrx/legacy`

Basic text extraction without AI features (for backward compatibility).

### Utility Endpoints

- **GET** `/` - System status and capabilities
- **GET** `/health` - Health check
- **GET** `/docs` - Interactive API documentation
- **GET** `/hackrx/vector-store/stats` - Vector store statistics

## 💡 Usage Examples

### PowerShell (Windows)

```powershell
# Simple question answering
curl -X POST "http://127.0.0.1:8000/hackrx/answers" -H "Content-Type: application/json" -d '{\"documents\": \"https://example.com/policy.pdf\", \"questions\": [\"What is the grace period?\", \"What are the coverage limits?\"]}'

# Search within document
curl -X POST "http://127.0.0.1:8000/hackrx/search?query=premium%20payment&top_k=3"
```

### Python

```python
import requests

# Process document and get answers
response = requests.post(
    "http://127.0.0.1:8000/hackrx/answers",
    json={
        "documents": "https://example.com/document.pdf",
        "questions": [
            "What is the main topic?",
            "What are the key findings?"
        ]
    }
)

if response.status_code == 200:
    answers = response.json()["answers"]
    for i, answer in enumerate(answers, 1):
        print(f"Answer {i}: {answer}")
```

## 🔧 Configuration

The system can be configured using environment variables in your `.env` file:

```env
# Required: Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here
LLM_MODEL=gemini-2.0-flash-exp

# Optional: Embedding Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Optional: Retrieval Configuration
TOP_K_CHUNKS=10
SIMILARITY_THRESHOLD=0.3

# Optional: Processing Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_FILE_SIZE=50000000
```

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │───▶│  Text Extraction │───▶│   Text Chunks   │
│   Download      │    │  (PDF/DOCX)      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Question      │───▶│   Embedding      │───▶│  Vector Store   │
│   Processing    │    │   Generation     │    │  (FAISS)        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                                               │
         ▼                                               ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gemini LLM    │◀───│   Context        │◀───│   Similarity    │
│   Response      │    │   Retrieval      │    │   Search        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🧠 AI Capabilities

### Advanced Question Types Supported

- **Factual Questions**: "What is the grace period for payments?"
- **Analytical Questions**: "What are the main differences between Plan A and Plan B?"
- **Comparative Questions**: "How does this policy compare to standard coverage?"
- **Summarization**: "What are the key benefits of this policy?"
- **Detail Extraction**: "What are all the waiting periods mentioned?"
- **Conditional Logic**: "Under what conditions is maternity covered?"

### Intelligent Features

- **Semantic Understanding**: Goes beyond keyword matching to understand context and meaning
- **Multi-chunk Reasoning**: Combines information from multiple document sections
- **Confidence Scoring**: Each answer includes a confidence score
- **Source Attribution**: Links answers back to specific document sections
- **Fallback Handling**: Graceful degradation when AI services are unavailable

## 📊 Performance

- **Processing Speed**: ~2-5 seconds for typical policy documents
- **Accuracy**: High accuracy on factual questions with proper source attribution
- **Scalability**: Handles documents up to 50MB
- **Memory Efficient**: Uses vector compression for large documents

## 🔍 Troubleshooting

### Common Issues

1. **"No relevant information found"**
   - Check similarity threshold in configuration
   - Ensure document was processed successfully
   - Try broader questions

2. **Slow processing**
   - Large documents take longer to process
   - Check network connectivity for document download

3. **API errors**
   - Verify Gemini API key is correct
   - Check rate limits on API usage

### Debug Endpoints

- **GET** `/hackrx/debug/chunks` - Inspect processed document chunks
- **GET** `/hackrx/vector-store/stats` - Check vector store status

## 📝 License

This project is developed for the Bajaj Hackathon challenge.

## 🤝 Support

For issues and questions, check the interactive API documentation at `http://127.0.0.1:8000/docs` when the server is running.

## 🚀 Quick Start

1. **Start the server:**
   ```bash
   python run.py
   ```

2. **Visit the interactive API docs:**
   Open your browser and go to `http://127.0.0.1:8000/docs`

3. **Process a document:**
   ```bash
   curl -X POST "http://127.0.0.1:8000/hackrx/run" \
        -H "Content-Type: application/json" \
        -d '{
          "documents": "https://example.com/document.pdf",
          "questions": ["What is the main topic?"]
        }'
   ```

## 📚 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root endpoint with system information |
| `GET` | `/health` | Health check endpoint |
| `POST` | `/hackrx/run` | Process document and extract chunks |
| `POST` | `/hackrx/validate` | Validate document URL |

### Documentation Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/docs` | Interactive API documentation (Swagger UI) |
| `GET` | `/redoc` | Alternative API documentation (ReDoc) |

## 💡 Usage Examples

### 1. Process a Document

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/hackrx/run" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
       "questions": [
         "What is the main topic of this document?",
         "What are the key findings?"
       ]
     }'
```

**Response:**
```json
{
  "success": true,
  "message": "Document processed successfully",
  "chunks": [
    {
      "chunk_id": "chunk_0000",
      "text": "Document content chunk...",
      "start_char": 0,
      "end_char": 500,
      "page_number": 1,
      "metadata": {
        "chunk_index": 0,
        "word_count": 75,
        "char_count": 500
      }
    }
  ],
  "metadata": {
    "request_id": "req_1234567890",
    "total_chunks": 15,
    "total_characters": 7500,
    "file_type": "pdf",
    "processing_time_seconds": 2.5
  }
}
```

### 2. Validate a Document URL

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/hackrx/validate" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "https://example.com/document.pdf",
       "questions": []
     }'
```

**Response:**
```json
{
  "valid": true,
  "url": "https://example.com/document.pdf",
  "message": "Document is accessible",
  "timestamp": "2025-07-26T10:30:00"
}
```

### 3. Using Python Requests

```python
import requests

# Process a document
response = requests.post(
    "http://127.0.0.1:8000/hackrx/run",
    json={
        "documents": "https://example.com/document.pdf",
        "questions": ["What is this document about?"]
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"Processed {len(result['chunks'])} chunks")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

## ⚙️ Configuration

### Environment Variables

You can configure the application using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `127.0.0.1` | Host address for the API server |
| `API_PORT` | `8000` | Port number for the API server |
| `DEBUG` | `False` | Enable debug mode |
| `MAX_FILE_SIZE` | `50000000` | Maximum file size in bytes (50MB) |
| `REQUEST_TIMEOUT` | `300` | Request timeout in seconds |
| `MAX_RETRIES` | `3` | Maximum retry attempts for downloads |
| `CHUNK_SIZE` | `1000` | Default chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks in characters |

### Configuration File

Settings are managed in `app/config/settings.py`:

```python
# Example configuration
API_HOST = "0.0.0.0"  # Listen on all interfaces
API_PORT = 8080       # Use port 8080
DEBUG = True          # Enable debug mode
CHUNK_SIZE = 1500     # Larger chunks
```

## 📁 Project Structure

```
BajajFinserv/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py         # Configuration settings
│   ├── core/
│   │   ├── __init__.py
│   │   └── document_processor.py  # Document processing logic
│   ├── models/
│   │   ├── __init__.py
│   │   ├── request_models.py   # Pydantic request models
│   │   └── response_models.py  # Pydantic response models
│   └── utils/
│       ├── __init__.py
│       ├── exceptions.py       # Custom exceptions
│       └── text_processing.py  # Text processing utilities
├── run.py                      # Application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── test_files/                 # Test scripts
    ├── comprehensive_test.py
    ├── debug_api.py
    └── simple_test.py
```

## 🧪 Testing

### Automated Tests

Run the comprehensive test suite:

```bash
python comprehensive_test.py
```

### Manual Testing

1. **Start the server:**
   ```bash
   python run.py
   ```

2. **Use the interactive docs:**
   Visit `http://127.0.0.1:8000/docs`

3. **Test with curl:**
   ```bash
   # Health check
   curl http://127.0.0.1:8000/health
   
   # Process document
   curl -X POST "http://127.0.0.1:8000/hackrx/run" \
        -H "Content-Type: application/json" \
        -d '{"documents": "https://example.com/test.pdf", "questions": []}'
   ```

### Debug Mode

For detailed error information, run the debug script:

```bash
python debug_api.py
```

## 🔧 Supported File Formats

- **PDF**: `.pdf` files
- **Microsoft Word**: `.docx`, `.doc` files

### Supported MIME Types

- `application/pdf`
- `application/vnd.openxmlformats-officedocument.wordprocessingml.document`
- `application/msword`

## 📊 Performance Considerations

- **File Size Limit**: 50MB by default (configurable)
- **Timeout**: 5 minutes for document processing
- **Retry Logic**: 3 retry attempts for failed downloads
- **Chunking**: Optimized for 1000-character chunks with 200-character overlap

## 🚨 Error Handling

The API provides detailed error responses:

### Common Error Codes

- `400 Bad Request`: Invalid input data
- `422 Unprocessable Entity`: Validation errors
- `500 Internal Server Error`: Processing failures

### Error Response Format

```json
{
  "detail": "Error description",
  "error_code": "DOCUMENT_PROCESSING_ERROR",
  "timestamp": "2025-07-26T10:30:00"
}
```

## 🔮 Future Phases

This is Phase 1 of the Intelligent Query-Retrieval System. Future phases will include:

- **Phase 2**: Document embedding and vector storage
- **Phase 3**: Question answering with LLM integration
- **Phase 4**: Advanced search and retrieval capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is part of the Bajaj Finserv Hackathon submission.

## 🆘 Support

If you encounter any issues:

1. Check the logs in the console output
2. Verify your document URL is accessible
3. Test with the `/health` endpoint
4. Review the API documentation at `/docs`

For debugging, use the provided test scripts:
- `comprehensive_test.py` - Full functionality test
- `debug_api.py` - Detailed error analysis
- `simple_test.py` - Basic connectivity test

---

**Built with ❤️ for the Bajaj Finserv Hackathon**
