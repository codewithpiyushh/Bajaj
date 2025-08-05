# app/core/llm_service.py
import logging
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import json

from app.config.settings import settings
from app.models.response_models import DocumentChunk

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service for generating answers using Gemini LLM based on retrieved context
    """
    
    def __init__(self):
        self.gemini_client = None
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.gemini_client = genai.GenerativeModel(settings.LLM_MODEL)
            logger.info("Gemini client initialized")
        else:
            logger.warning("No Gemini API key provided. Using fallback responses.")
    
    def _create_context_from_chunks(self, chunks: List[Tuple[DocumentChunk, float]]) -> str:
        """
        Create context string from retrieved chunks
        
        Args:
            chunks: List of (chunk, similarity_score) tuples
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, (chunk, score) in enumerate(chunks, 1):
            context_parts.append(f"Context {i} (relevance: {score:.2f}):\n{chunk.text}\n")
        
        return "\n".join(context_parts)
    
    def _clean_and_shorten_answer(self, answer: str) -> str:
        """
        Clean and shorten the answer to ensure it's concise
        
        Args:
            answer: Raw answer from LLM
            
        Returns:
            Cleaned and shortened answer
        """
        # Remove common verbose phrases
        verbose_phrases = [
            "based on the information provided",
            "according to the document",
            "the document states",
            "as mentioned in the context",
            "from the information available",
            "the policy states that",
            "it is mentioned that",
        ]
        
        answer = answer.strip()
        
        # Remove verbose phrases
        for phrase in verbose_phrases:
            answer = answer.replace(phrase, "").replace(phrase.title(), "")
        
        # Take only the first sentence if multiple sentences
        sentences = answer.split('.')
        if len(sentences) > 1 and len(sentences[0].strip()) > 5:
            answer = sentences[0].strip() + '.'
        
        # Limit to reasonable length (around 15-20 words max)
        words = answer.split()
        if len(words) > 20:
            answer = ' '.join(words[:20]) + '.'
        
        # Clean up multiple spaces and formatting
        answer = ' '.join(answer.split())
        
        return answer
    
    def _create_prompt(self, question: str, context: str) -> str:
        """
        Create a prompt for the LLM to generate detailed, comprehensive answers
        
        Args:
            question: User's question
            context: Retrieved context from document chunks
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert document analyst who provides comprehensive, detailed answers based on policy documents.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide a complete, detailed answer based ONLY on the information in the document context above
- Include ALL relevant details, conditions, timeframes, and specifics mentioned in the document
- Use the exact terminology and phrases from the document when possible
- If there are specific numbers, timeframes, percentages, or conditions, include them all
- Write in complete sentences with proper explanations
- Do NOT add information not found in the context
- If the document doesn't contain the information, say "The document does not provide information about this topic"

Provide your comprehensive answer:"""
        
        return prompt
    
    async def generate_answer_gemini(self, question: str, context: str) -> Dict[str, Any]:
        """
        Generate answer using Gemini API
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self.gemini_client:
            raise ValueError("Gemini client not initialized")
        
        prompt = self._create_prompt(question, context)
        
        try:
            # Use asyncio.to_thread for non-async Gemini calls
            response = await asyncio.to_thread(
                self.gemini_client.generate_content,
                prompt,
                generation_config={
                    "temperature": 0.1,  # Low temperature for consistent answers
                    "max_output_tokens": 200,  # Increased for detailed answers
                    "top_p": 0.9,  # Allow more diverse responses
                }
            )
            
            answer = response.text
            
            # Clean up the answer but keep it detailed
            answer = answer.strip()
            
            # Remove any quotation marks that might wrap the answer
            if answer.startswith('"') and answer.endswith('"'):
                answer = answer[1:-1]
            
            # Ensure it ends with proper punctuation
            if answer and not answer.endswith(('.', '!', '?')):
                answer += '.'
            
            return {
                "answer": answer,
                "model": settings.LLM_MODEL,
                "tokens_used": len(prompt.split()) + len(answer.split()),  # Approximate
                "method": "gemini"
            }
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            # Fallback to simple response
            return await self.generate_answer_fallback(question, context)
    
    async def generate_answer_fallback(self, question: str, context: str) -> Dict[str, Any]:
        """
        Generate a simple fallback answer when LLM is not available
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            Dictionary with answer and metadata
        """
        # Simple keyword-based approach for fallback
        question_lower = question.lower()
        context_lower = context.lower()
        
        # Look for key question words in context
        key_words = []
        for word in question_lower.split():
            if len(word) > 3 and word in context_lower:
                key_words.append(word)
        
        if key_words:
            # Extract sentences containing key words
            context_sentences = context.split('.')
            relevant_sentences = []
            for sentence in context_sentences[:5]:  # Limit to first 5 sentences
                if any(word in sentence.lower() for word in key_words):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                answer = " ".join(relevant_sentences[:2])  # Use top 2 relevant sentences
                # Clean up the answer
                answer = answer.strip()
                if not answer.endswith('.'):
                    answer += '.'
            else:
                answer = "Information not available in the document."
        else:
            answer = "Information not available in the document."
        
        return {
            "answer": answer,
            "model": "fallback",
            "tokens_used": 0,
            "method": "keyword_matching"
        }
    
    async def answer_question(
        self, 
        question: str, 
        retrieved_chunks: List[Tuple[DocumentChunk, float]]
    ) -> Dict[str, Any]:
        """
        Main method to answer a question using retrieved context
        
        Args:
            question: User's question
            retrieved_chunks: List of (chunk, similarity_score) tuples from vector search
            
        Returns:
            Dictionary containing answer and metadata
        """
        # Create context from retrieved chunks
        context = self._create_context_from_chunks(retrieved_chunks)
        
        # Generate answer
        if self.gemini_client and settings.GEMINI_API_KEY:
            try:
                result = await self.generate_answer_gemini(question, context)
            except Exception as e:
                logger.warning(f"Gemini failed, using fallback: {e}")
                result = await self.generate_answer_fallback(question, context)
        else:
            result = await self.generate_answer_fallback(question, context)
        
        # Add context metadata
        result.update({
            "context_chunks_used": len(retrieved_chunks),
            "context_preview": context[:200] + "..." if len(context) > 200 else context,
            "chunk_sources": [
                {
                    "chunk_id": chunk.chunk_id,
                    "similarity": float(score),
                    "page_number": chunk.page_number
                } 
                for chunk, score in retrieved_chunks
            ]
        })
        
        return result
    
    async def answer_multiple_questions(
        self,
        questions: List[str],
        retrieved_chunks_per_question: List[List[Tuple[DocumentChunk, float]]]
    ) -> List[Dict[str, Any]]:
        """
        Answer multiple questions
        
        Args:
            questions: List of questions
            retrieved_chunks_per_question: List of retrieved chunks for each question
            
        Returns:
            List of answer dictionaries
        """
        if len(questions) != len(retrieved_chunks_per_question):
            raise ValueError("Number of questions must match number of chunk lists")
        
        tasks = []
        for question, chunks in zip(questions, retrieved_chunks_per_question):
            task = self.answer_question(question, chunks)
            tasks.append(task)
        
        # Process all questions concurrently
        answers = await asyncio.gather(*tasks)
        
        return answers
