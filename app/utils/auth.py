# app/utils/auth.py
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Configure your API token here
API_TOKEN = "c18bfd661dbe0c511a945fef2967b5a1fdf7246996e89e41408bafed5c7b5465"

security = HTTPBearer()

def verify_token(authorization: Optional[str]) -> bool:
    """
    Verify Bearer token authentication
    
    Args:
        authorization: Authorization header value
        
    Returns:
        bool: True if token is valid, False otherwise
    """
    if not authorization:
        logger.warning("No authorization header provided")
        return False
    
    try:
        # Extract token from "Bearer <token>"
        if not authorization.startswith("Bearer "):
            logger.warning("Invalid authorization format")
            return False
        
        token = authorization.replace("Bearer ", "").strip()
        
        # Verify token
        if token == API_TOKEN:
            logger.info("Valid token provided")
            return True
        else:
            logger.warning("Invalid token provided")
            return False
            
    except Exception as e:
        logger.error(f"Error verifying token: {e}")
        return False

def require_auth(authorization: Optional[str]):
    """
    Require authentication for protected endpoints
    
    Args:
        authorization: Authorization header value
        
    Raises:
        HTTPException: If authentication fails
    """
    if not verify_token(authorization):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authorization token",
            headers={"WWW-Authenticate": "Bearer"},
        ) 