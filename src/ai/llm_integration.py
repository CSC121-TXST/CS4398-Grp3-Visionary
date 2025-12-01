"""
LLM Integration Module

This module provides an abstraction layer for communicating with
Large Language Model APIs (OpenAI, HuggingFace, etc.)
"""

import os
import time
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
# Try to load from src/ directory first (where user placed .env), then root
env_paths = [
    Path(__file__).parent.parent / ".env",  # src/.env
    Path(__file__).parent.parent.parent / ".env",  # root/.env
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break
else:
    # Fallback to default behavior (searches from current working directory)
    load_dotenv()

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed. LLM features will be disabled.")


class LLMClient:
    """
    Client for interacting with LLM APIs.
    
    Currently supports OpenAI API. Can be extended for other providers.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 150,
        temperature: float = 0.7,
    ):
        """
        Initialize LLM client.
        
        Args:
            provider: LLM provider ("openai", "huggingface", etc.)
            api_key: API key for the provider (if None, reads from env)
            model: Model name to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-2.0)
        """
        self.provider = provider.lower()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize provider-specific client
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package required for OpenAI provider")
            if not self.api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
            self.client = OpenAI(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def generate_description(
        self,
        detection_summary: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a natural language description from detection data.
        
        Args:
            detection_summary: Formatted detection data (e.g., "2 persons, 1 laptop")
            system_prompt: Optional custom system prompt
            
        Returns:
            Natural language description string
            
        Raises:
            Exception: If API call fails
        """
        if self.provider == "openai":
            return self._generate_openai(detection_summary, system_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _generate_openai(
        self,
        detection_summary: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate description using OpenAI API."""
        default_system_prompt = (
            "You are an accessibility assistant. Convert detection data into "
            "a short, natural, spoken-style description. Be concise and clear. "
            "Use first person (e.g., 'I see...')."
        )
        
        system_prompt = system_prompt or default_system_prompt
        
        user_prompt = f"Detections: {detection_summary}\n\nProvide a natural description of what is visible."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if LLM client is properly configured and available."""
        try:
            if self.provider == "openai":
                return OPENAI_AVAILABLE and self.api_key is not None
            return False
        except Exception:
            return False


class MockLLMClient:
    """
    Mock LLM client for testing without API calls.
    Returns simple formatted descriptions.
    """
    
    def __init__(self, *args, **kwargs):
        pass
    
    def generate_description(self, detection_summary: str, system_prompt: Optional[str] = None) -> str:
        """Generate a simple mock description."""
        return f"I see {detection_summary.lower()} in the frame."
    
    def is_available(self) -> bool:
        return True

