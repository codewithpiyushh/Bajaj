# app/utils/text_processing.py
import re
import string
from typing import List, Dict, Any
import math

class TextProcessor:
    """Handles text cleaning and chunking operations"""
    
    def __init__(self):
        self.sentence_endings = r'[.!?]+\s+'
        self.paragraph_breaks = r'\n\s*\n'
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove non-printable characters except newlines and tabs
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        # Normalize quotation marks (fix Unicode characters)
        text = re.sub(r'["\u201C\u201D]', '"', text)  # Replace smart quotes with regular quotes
        text = re.sub(r"['\u2018\u2019]", "'", text)  # Replace smart apostrophes with regular apostrophes
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def split_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks for processing
        
        Args:
            text: The text to split
            chunk_size: Target size for each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of dictionaries containing chunk information
        """
        if not text:
            return []
        
        chunks = []
        text_length = len(text)
        
        if text_length <= chunk_size:
            # Text is small enough to be a single chunk
            return [{
                'text': text,
                'start': 0,
                'end': text_length,
                'page_number': None
            }]
        
        start = 0
        chunk_index = 0
        
        while start < text_length:
            # Calculate end position for this chunk
            end = min(start + chunk_size, text_length)
            
            # If we're not at the end of the text, try to break at a natural boundary
            if end < text_length:
                # Look for sentence boundaries within the last 200 characters
                search_start = max(end - 200, start)
                sentence_match = None
                
                # Find the last sentence ending before our target end
                for match in re.finditer(self.sentence_endings, text[search_start:end]):
                    sentence_match = match
                
                if sentence_match:
                    # Adjust end to the sentence boundary
                    end = search_start + sentence_match.end()
                else:
                    # Look for word boundaries
                    while end > start and not text[end-1].isspace():
                        end -= 1
                    
                    # If we couldn't find a word boundary, just use the original end
                    if end == start:
                        end = min(start + chunk_size, text_length)
            
            # Extract the chunk text
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunks.append({
                    'text': chunk_text,
                    'start': start,
                    'end': end,
                    'page_number': None  # Page numbers would need to be tracked separately
                })
                chunk_index += 1
            
            # Calculate next start position with overlap
            if end >= text_length:
                break
                
            # Move start position, considering overlap
            next_start = end - overlap
            
            # Ensure we don't go backwards
            if next_start <= start:
                next_start = start + 1
                
            start = next_start
        
        return chunks
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract potential keywords from text (simple implementation)
        """
        if not text:
            return []
        
        # Convert to lowercase and remove punctuation
        text_clean = text.lower()
        text_clean = text_clean.translate(str.maketrans('', '', string.punctuation))
        
        # Split into words
        words = text_clean.split()
        
        # Common stop words to filter out
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'under', 'over', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Filter words and count frequency
        word_freq = {}
        for word in words:
            if len(word) > 2 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in keywords[:max_keywords]]
    
    def get_text_stats(self, text: str) -> Dict[str, Any]:
        """
        Get basic statistics about the text
        """
        if not text:
            return {
                'char_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'paragraph_count': 0
            }
        
        # Character count
        char_count = len(text)
        
        # Word count
        words = text.split()
        word_count = len(words)
        
        # Sentence count (rough estimate)
        sentences = re.split(self.sentence_endings, text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Paragraph count
        paragraphs = re.split(self.paragraph_breaks, text)
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count
        }
