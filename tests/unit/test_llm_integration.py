"""
Unit tests for LLM Integration component.
"""

import pytest
import os
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ai.llm_integration import LLMClient


@pytest.mark.unit
@pytest.mark.ai
class TestLLMClient:
    """Test suite for LLMClient class."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-12345"})
    @patch('ai.llm_integration.OpenAI')
    def test_initialization_with_env_key(self, mock_openai_class):
        """Test LLMClient initialization with environment variable."""
        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance
        
        client = LLMClient()
        assert client is not None
        assert client.api_key == "sk-test-key-12345"
        assert client.provider == "openai"
        mock_openai_class.assert_called_once_with(api_key="sk-test-key-12345")
    
    @patch('ai.llm_integration.OpenAI')
    def test_initialization_with_provided_key(self, mock_openai_class):
        """Test LLMClient initialization with provided API key."""
        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance
        
        client = LLMClient(api_key="sk-provided-key")
        assert client.api_key == "sk-provided-key"
        mock_openai_class.assert_called_once_with(api_key="sk-provided-key")
    
    @patch.dict(os.environ, {}, clear=True)
    def test_initialization_without_key(self):
        """Test that initialization fails without API key."""
        with pytest.raises(ValueError, match="OpenAI API key required"):
            LLMClient()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    @patch('ai.llm_integration.OpenAI')
    def test_generate_description(self, mock_openai_class):
        """Test description generation."""
        # Setup mock
        mock_client_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I see two people in the frame."
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client_instance
        
        client = LLMClient()
        result = client.generate_description("2 persons, 1 laptop")
        
        assert result == "I see two people in the frame."
        mock_client_instance.chat.completions.create.assert_called_once()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    @patch('ai.llm_integration.OpenAI')
    def test_generate_description_custom_prompt(self, mock_openai_class):
        """Test description generation with custom system prompt."""
        mock_client_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Custom description"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client_instance
        
        client = LLMClient()
        custom_prompt = "You are a helpful assistant."
        result = client.generate_description("test", system_prompt=custom_prompt)
        
        assert result == "Custom description"
        call_args = mock_client_instance.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == custom_prompt
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    @patch('ai.llm_integration.OpenAI')
    def test_generate_description_api_error(self, mock_openai_class):
        """Test error handling when API call fails."""
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client_instance
        
        client = LLMClient()
        with pytest.raises(Exception, match="OpenAI API error"):
            client.generate_description("test")
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    @patch('ai.llm_integration.OpenAI')
    def test_is_available(self, mock_openai_class):
        """Test availability check."""
        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance
        
        client = LLMClient()
        assert client.is_available() == True
    
    @patch.dict(os.environ, {}, clear=True)
    def test_is_available_no_key(self):
        """Test availability check without API key."""
        with pytest.raises(ValueError):
            LLMClient()

