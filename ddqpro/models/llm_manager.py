from typing import Optional
from langchain.chat_models.base import BaseChatModel
from .llm_factory import LLMFactory  # Add this import


class LLMManager:
    _instance = None
    _llm: Optional[BaseChatModel] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
        return cls._instance

    def initialize(self, provider: str, model_name: str, **kwargs):
        """Initialize the LLM with specified provider and model"""
        print(f"Initializing LLM with provider: {provider}, model: {model_name}")  # Debug print

        factory = LLMFactory()
        self._llm = factory.create_llm(provider, model_name, **kwargs)

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            raise RuntimeError("LLM not initialized. Call initialize() first.")
        return self._llm