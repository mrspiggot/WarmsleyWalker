# src/models/llm_factory.py

from typing import Dict, Type
from abc import ABC, abstractmethod
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import logging
logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    @abstractmethod
    def create_llm(self, model_name: str, **kwargs) -> BaseChatModel:
        """Create and return an LLM instance"""
        pass


class OpenAIProvider(LLMProvider):
    def create_llm(self, model_name: str, **kwargs) -> BaseChatModel:
        return ChatOpenAI(model_name=model_name, **kwargs)


class AnthropicProvider(LLMProvider):
    def create_llm(self, model_name: str, **kwargs) -> BaseChatModel:
        return ChatAnthropic(model_name=model_name, **kwargs)


class GoogleProvider(LLMProvider):
    def create_llm(self, model_name: str, **kwargs) -> BaseChatModel:
        return ChatGoogleGenerativeAI(model=model_name, **kwargs)


class OllamaProvider(LLMProvider):
    def __init__(self):
        self.available_models = [
            "llama2",
            "TiagoG/json-response",
            "mixtral:8x22b",
            "llama3.1:70b"
        ]
        self._current_model_idx = 0

    def create_llm(self, model_name: str = None, **kwargs) -> BaseChatModel:
        """Create LLM with fallback options"""
        if model_name:
            try:
                self._current_model_idx = self.available_models.index(model_name)
            except ValueError:
                logger.warning(f"Model {model_name} not found in available models")

        while self._current_model_idx < len(self.available_models):
            try:
                model = self.available_models[self._current_model_idx]
                logger.info(f"Trying Ollama model: {model}")

                # Remove duplicate parameters
                kwargs.pop('temperature', None)  # Remove if present in kwargs
                kwargs.pop('format', None)  # Remove if present in kwargs

                return ChatOllama(
                    model=model,
                    format="json",
                    temperature=0.0,
                    **kwargs
                )
            except Exception as e:
                logger.warning(f"Failed to load {model}: {str(e)}")
                self._current_model_idx += 1

        raise ValueError("No available Ollama models")


class LLMFactory:
    def __init__(self):
        self.providers: Dict[str, Type[LLMProvider]] = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "google": GoogleProvider(),
            "ollama": OllamaProvider()
        }

    def create_llm(self, provider: str, model_name: str, **kwargs) -> BaseChatModel:
        provider_lower = provider.lower()
        if provider_lower not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}")

        return self.providers[provider_lower].create_llm(model_name, **kwargs)


# src/models/llm_manager.py

