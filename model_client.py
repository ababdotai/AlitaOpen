#!/usr/bin/env python3
"""
This module implements a unified model client interface that supports both OpenAI and Anthropic models.
It provides a factory pattern to create appropriate clients based on configuration and abstracts
the differences between different model providers.

The ModelClientFactory creates either OpenAI or Anthropic clients based on the model name,
while the unified interface ensures consistent API calls across different providers.
"""

import os
import logging
from typing import Any, Dict, List
from abc import ABC, abstractmethod

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from utils import AlitaError


class ModelClient(ABC):
    """
    Abstract base class for model clients that provides a unified interface
    for different LLM providers (OpenAI, Anthropic, etc.).
    """
    
    @abstractmethod
    def create_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Create a completion using the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional parameters like temperature, max_tokens, etc.
            
        Returns:
            str: The completion text from the model
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the model name being used.
        
        Returns:
            str: The model name
        """
        pass


class OpenAIClient(ModelClient):
    """
    OpenAI model client implementation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI client with configuration.
        
        Args:
            config: Configuration dictionary containing API settings
        """
        if OpenAI is None:
            raise AlitaError("OpenAI library not installed. Please install with: pip install openai")
            
        api_config = config.get("api", {})
        self.model = config.get("agent", {}).get("primary_llm", "gpt-4o")
        self.temperature = float(api_config.get("temperature", 0.7))
        self.max_tokens = int(api_config.get("max_tokens", 16384))
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_config.get("openai_api_key", os.environ.get("OPENAI_API_KEY")),
            base_url=api_config.get("openai_api_url", "https://api.openai.com/v1")
        )
        
        logging.info("OpenAI client initialized with model: %s", self.model)
    
    def create_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Create a completion using OpenAI's API.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            str: The completion text
        """
        try:
            # Use provided parameters or fall back to defaults
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error("OpenAI API call failed: %s", str(e))
            raise AlitaError(f"OpenAI API call failed: {str(e)}") from e
    
    def get_model_name(self) -> str:
        """
        Get the OpenAI model name.
        
        Returns:
            str: The model name
        """
        return self.model


class AnthropicClient(ModelClient):
    """
    Anthropic model client implementation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Anthropic client with configuration.
        
        Args:
            config: Configuration dictionary containing API settings
        """
        if Anthropic is None:
            raise AlitaError("Anthropic library not installed. Please install with: pip install anthropic")
            
        api_config = config.get("api", {})
        self.model = config.get("agent", {}).get("primary_llm", "claude-3-5-sonnet-latest")
        self.temperature = float(api_config.get("temperature", 0.7))
        self.max_tokens = int(api_config.get("max_tokens", 16384))
        
        # Initialize Anthropic client
        self.client = Anthropic(
            api_key=api_config.get("anthropic_api_key", os.environ.get("ANTHROPIC_API_KEY")),
            base_url=api_config.get("anthropic_base_url", "https://api.anthropic.com/v1")
        )
        
        logging.info("Anthropic client initialized with model: %s", self.model)
    
    def create_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Create a completion using Anthropic's API.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            str: The completion text
        """
        try:
            # Use provided parameters or fall back to defaults
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            
            response = self.client.messages.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract text content from Anthropic response
            if response.content and len(response.content) > 0:
                return response.content[0].text
            else:
                return ""
                
        except Exception as e:
            logging.error("Anthropic API call failed: %s", str(e))
            raise AlitaError(f"Anthropic API call failed: {str(e)}") from e
    
    def get_model_name(self) -> str:
        """
        Get the Anthropic model name.
        
        Returns:
            str: The model name
        """
        return self.model


class ModelClientFactory:
    """
    Factory class for creating appropriate model clients based on configuration.
    """
    
    # Define which models belong to which providers
    OPENAI_MODELS = {
        # GPT-4o series
        "gpt-4o", "gpt-4o-mini", "gpt-4o-2024-11-20", "gpt-4o-2024-08-06", "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18", "gpt-4o-realtime-preview", "gpt-4o-realtime-preview-2024-10-01",
        "gpt-4o-audio-preview", "gpt-4o-audio-preview-2024-10-01",
        # GPT-4 series
        "gpt-4", "gpt-4-turbo", "gpt-4-turbo-2024-04-09", "gpt-4-turbo-preview",
        "gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4-vision-preview",
        "gpt-4-1106-vision-preview", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613",
        # GPT-3.5 series
        "gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-instruct",
        # o1 series (reasoning models)
        "o1", "o1-preview", "o1-preview-2024-09-12", "o1-mini", "o1-mini-2024-09-12"
    }
    
    ANTHROPIC_MODELS = {
        # Claude 4 series
        "claude-opus-4-20250514", "claude-sonnet-4-20250514",
        "claude-opus-4-0", "claude-sonnet-4-0",
        # Claude 3.7 series
        "claude-3-7-sonnet-20250219", "claude-3-7-sonnet-latest",
        # Claude 3.5 series
        "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-latest", "claude-3-5-sonnet-20240620",
        "claude-3-5-haiku-20241022", "claude-3-5-haiku-latest",
        # Claude 3 series
        "claude-3-opus-20240229", "claude-3-opus-latest",
        "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
    }
    
    @classmethod
    def create_client(cls, config: Dict[str, Any]) -> ModelClient:
        """
        Create an appropriate model client based on the configured model.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            ModelClient: An instance of the appropriate model client
            
        Raises:
            AlitaError: If the model is not supported or configuration is invalid
        """
        model_name = config.get("agent", {}).get("primary_llm", "")
        
        if not model_name:
            raise AlitaError("No model specified in configuration under agent.primary_llm")
        
        # Determine provider based on model name
        if model_name in cls.OPENAI_MODELS:
            return OpenAIClient(config)
        elif model_name in cls.ANTHROPIC_MODELS:
            return AnthropicClient(config)
        else:
            # Try to infer from model name patterns
            if "gpt" in model_name.lower():
                logging.warning("Unknown OpenAI model '%s', attempting to use OpenAI client", model_name)
                return OpenAIClient(config)
            elif "claude" in model_name.lower():
                logging.warning("Unknown Anthropic model '%s', attempting to use Anthropic client", model_name)
                return AnthropicClient(config)
            else:
                raise AlitaError(f"Unsupported model: {model_name}. Supported providers: OpenAI, Anthropic")
    
    @classmethod
    def get_supported_models(cls) -> Dict[str, List[str]]:
        """
        Get a dictionary of supported models by provider.
        
        Returns:
            Dict[str, List[str]]: Dictionary with provider names as keys and model lists as values
        """
        return {
            "openai": list(cls.OPENAI_MODELS),
            "anthropic": list(cls.ANTHROPIC_MODELS)
        }