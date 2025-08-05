#!/bin/bash

# Intelligent Document Query System - API Test Script
# Make sure your API is running on http://localhost:8000

echo "🧪 Testing Intelligent Document Query System API"
echo "================================================"

# Test 1: Health Check
echo ""
echo "1️⃣ Testing Health Check..."
curl -X GET "http://localhost:8000/health" \
  -H "Content-Type: application/json" \
  -w "\nStatus: %{http_code}\nTime: %{time_total}s\n"

# Test 2: Root Endpoint
echo ""
echo "2️⃣ Testing Root Endpoint..."
curl -X GET "http://localhost:8000/" \
  -H "Content-Type: application/json" \
  -w "\nStatus: %{http_code}\nTime: %{time_total}s\n"

# Test 3: Document Processing with Q&A (Main Endpoint)
echo ""
echo "3️⃣ Testing Document Processing with Q&A..."
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    "questions": [
      "What is the main topic of this document?",
      "What are the key benefits mentioned?",
      "What is the grace period for claims?"
    ]
  }' \
  -w "\nStatus: %{http_code}\nTime: %{time_total}s\n"

# Test 4: Simple Answers Endpoint
echo ""
echo "4️⃣ Testing Simple Answers Endpoint..."
curl -X POST "http://localhost:8000/hackrx/answers" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    "questions": [
      "What is the main topic?",
      "What are the coverage limits?"
    ]
  }' \
  -w "\nStatus: %{http_code}\nTime: %{time_total}s\n"

# Test 5: Document Validation
echo ""
echo "5️⃣ Testing Document Validation..."
curl -X POST "http://localhost:8000/hackrx/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    "questions": []
  }' \
  -w "\nStatus: %{http_code}\nTime: %{time_total}s\n"

# Test 6: Vector Store Stats
echo ""
echo "6️⃣ Testing Vector Store Stats..."
curl -X GET "http://localhost:8000/hackrx/vector-store/stats" \
  -H "Content-Type: application/json" \
  -w "\nStatus: %{http_code}\nTime: %{time_total}s\n"

echo ""
echo "✅ API Testing Complete!"
echo "📖 Interactive API docs: http://localhost:8000/docs"
echo "📋 ReDoc documentation: http://localhost:8000/redoc" 