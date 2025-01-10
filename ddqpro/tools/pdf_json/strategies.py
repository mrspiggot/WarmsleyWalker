import os
import json
from typing import List, Union, Optional
from langchain.schema import BaseMessage, SystemMessage, HumanMessage

import pdfplumber
from langchain.chat_models import ChatOpenAI  # or your chosen LLM library

##############################################################################
# OPTIONAL: Strategy Interface
##############################################################################

from abc import ABC, abstractmethod


class PDFExtractionStrategy(ABC):
    """
    Abstract base class that defines how we prompt the LLM
    and parse its output into JSON.
    """

    @abstractmethod
    def create_prompt_messages(self, pdf_content: str) -> List[dict]:
        """
        Return a list of message dicts with roles (e.g. 'system', 'user') that
        can be passed to a Chat model.
        """
        pass

    @abstractmethod
    def parse_llm_output(self, llm_response: str) -> str:
        """
        Given the raw string content from the LLM,
        produce a final JSON string or best-effort JSON.
        """
        pass


class BasicJSONStrategy(PDFExtractionStrategy):
    """
    A simple default strategy:
    - Provide a system prompt instructing the LLM to produce valid JSON
    - Provide a user prompt with the raw PDF text
    - Attempt to parse JSON
    """

    def create_prompt_messages(self, pdf_content: str) -> List[BaseMessage]:
        """
        Instead of returning a list of dictionaries, we now return
        [SystemMessage(...), HumanMessage(...)] which matches the
        expected interface for predict_messages().
        """
        system_prompt = (
            "You are a helpful assistant that converts PDF content into JSON. "
            "Please read the text below and produce a valid JSON representation, "
            "including the key information present in the PDF. Keep the structure "
            "simple but thorough."
        )
        user_prompt = f"Convert this PDF content to JSON:\n\n{pdf_content}"

        # Return actual LangChain message objects, not dicts:
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

    def parse_llm_output(self, llm_response: str) -> str:
        """
        Try to interpret llm_response as JSON. If it fails, return raw string.
        """
        try:
            data = json.loads(llm_response)
            # Return pretty-printed JSON
            return json.dumps(data, indent=2)
        except json.JSONDecodeError:
            # If not valid JSON, fallback
            return llm_response
