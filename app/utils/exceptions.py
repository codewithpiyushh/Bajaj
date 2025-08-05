# app/utils/exceptions.py

class DocumentProcessingError(Exception):
    """Custom exception for document processing errors"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

class DocumentDownloadError(DocumentProcessingError):
    """Exception raised when document download fails"""
    pass

class DocumentExtractionError(DocumentProcessingError):
    """Exception raised when text extraction fails"""
    pass

class DocumentValidationError(DocumentProcessingError):
    """Exception raised when document validation fails"""
    pass
