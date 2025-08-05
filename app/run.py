#!/usr/bin/env python3
"""
Entry point script for the Intelligent Query-Retrieval System
Run this script from the project root directory
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now we can import and run the app
if __name__ == "__main__":
    import uvicorn
    from app.config.settings import settings
    
    print(f"Starting Intelligent Query-Retrieval System...")
    print(f"API will be available at: http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"Interactive API docs: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print(f"ReDoc documentation: http://{settings.API_HOST}:{settings.API_PORT}/redoc")
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
