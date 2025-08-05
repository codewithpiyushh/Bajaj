@echo off
REM Intelligent Document Query System - API Test Script for Windows CMD
REM Make sure your API is running on http://localhost:8000

echo 🧪 Testing Intelligent Document Query System API
echo ================================================

echo.
echo 1️⃣ Testing Health Check...
curl -X GET "http://localhost:8000/health" -H "Content-Type: application/json"

echo.
echo 2️⃣ Testing Root Endpoint...
curl -X GET "http://localhost:8000/" -H "Content-Type: application/json"

echo.
echo 3️⃣ Testing Document Processing with Q&A...
curl -X POST "http://localhost:8000/hackrx/run" -H "Content-Type: application/json" -d "{\"documents\": \"https://hackrx.blob.core.windows.net/assets/Arogya%%20Sanjeevani%%20Policy%%20-%%20CIN%%20-%%20U10200WB1906GOI001713%%201.pdf?sv=2023-01-03^&st=2025-07-21T08%%3A29%%3A02Z^&se=2025-09-22T08%%3A29%%3A00Z^&sr=b^&sp=r^&sig=nzrz1K9Iurt%%2BBXom%%2FB%%2BMPTFMFP3PRnIvEsipAX10Ig4%%3D\", \"questions\": [\"What is the main topic of this document?\", \"What are the key benefits mentioned?\", \"What is the grace period for claims?\"]}"

echo.
echo 4️⃣ Testing Simple Answers Endpoint...
curl -X POST "http://localhost:8000/hackrx/answers" -H "Content-Type: application/json" -d "{\"documents\": \"https://hackrx.blob.core.windows.net/assets/Arogya%%20Sanjeevani%%20Policy%%20-%%20CIN%%20-%%20U10200WB1906GOI001713%%201.pdf?sv=2023-01-03^&st=2025-07-21T08%%3A29%%3A02Z^&se=2025-09-22T08%%3A29%%3A00Z^&sr=b^&sp=r^&sig=nzrz1K9Iurt%%2BBXom%%2FB%%2BMPTFMFP3PRnIvEsipAX10Ig4%%3D\", \"questions\": [\"What is the main topic?\", \"What are the coverage limits?\"]}"

echo.
echo 5️⃣ Testing Document Validation...
curl -X POST "http://localhost:8000/hackrx/validate" -H "Content-Type: application/json" -d "{\"documents\": \"https://hackrx.blob.core.windows.net/assets/Arogya%%20Sanjeevani%%20Policy%%20-%%20CIN%%20-%%20U10200WB1906GOI001713%%201.pdf?sv=2023-01-03^&st=2025-07-21T08%%3A29%%3A02Z^&se=2025-09-22T08%%3A29%%3A00Z^&sr=b^&sp=r^&sig=nzrz1K9Iurt%%2BBXom%%2FB%%2BMPTFMFP3PRnIvEsipAX10Ig4%%3D\", \"questions\": []}"

echo.
echo 6️⃣ Testing Vector Store Stats...
curl -X GET "http://localhost:8000/hackrx/vector-store/stats" -H "Content-Type: application/json"

echo.
echo ✅ API Testing Complete!
echo 📖 Interactive API docs: http://localhost:8000/docs
echo 📋 ReDoc documentation: http://localhost:8000/redoc
pause 