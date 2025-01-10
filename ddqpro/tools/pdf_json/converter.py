# File: /Users/richardwalker/PycharmProjects/WarmsleyWalker2025/ddqpro/tools/pdf_json/converter.py

import os
import json
from typing import List, Union, Optional

import pdfplumber
from langchain.chat_models import ChatOpenAI  # or your chosen LLM library

from ddqpro.tools.pdf_json.strategies import PDFExtractionStrategy, BasicJSONStrategy


class PDFtoJSONConverter:
    """
    A class to convert PDF files to JSON-formatted strings using an LLM.

    Parameters:
    -----------
    llm_model_name: str
        The name of the OpenAI model to use. Example: "gpt-3.5-turbo" or "gpt-4".
    openai_api_key: str
        API key for OpenAI.
    temperature: float
        Temperature setting for the LLM call.
    strategy: PDFExtractionStrategy
        Strategy object controlling how we create the prompt and parse the LLM output.
        If not supplied, defaults to BasicJSONStrategy.
    """

    def __init__(
        self,
        llm_model_name: str = "gpt-4",
        openai_api_key: Optional[str] = None,
        temperature: float = 0.0,
        strategy: Optional[PDFExtractionStrategy] = None,
    ):
        self.llm_model_name = llm_model_name
        self.openai_api_key = openai_api_key
        self.temperature = temperature

        # If strategy not provided, use a default
        self.strategy = strategy or BasicJSONStrategy()

        # Initialize the LLM
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """
        Private method to create an LLM client based on the specified model name.
        If you wish to adapt for Claude or Google, you can add logic here.
        """
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for GPT-based models.")

        # Using ChatOpenAI from langchain
        return ChatOpenAI(
            model_name=self.llm_model_name,
            openai_api_key=self.openai_api_key,
            temperature=self.temperature
        )

    def _split_into_chunks(self, text: str, max_chars: int = 3000) -> List[str]:
        """
        Naive approach: slice the text into chunks of up to `max_chars` length
        so we don't exceed the model's context window.

        You may want a more advanced approach (e.g., splitting by paragraphs
        or sentences) depending on your PDF content.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars
            chunks.append(text[start:end])
            start = end
        return chunks

    def _convert_chunks_to_json_list(self, text_chunks: List[str]) -> List[str]:
        """
        Convert each chunk to a JSON string (or best-effort JSON).
        Then you can either store them individually, or unify them yourself.
        """
        json_results = []
        for idx, chunk in enumerate(text_chunks, 1):
            prompt_messages = self.strategy.create_prompt_messages(chunk)
            response = self.llm.predict_messages(prompt_messages)
            chunk_json = self.strategy.parse_llm_output(response.content)
            json_results.append(chunk_json)
        return json_results

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a single PDF file using pdfplumber.
        """
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        text_chunks = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_chunks.append(page_text)
        return "\n".join(text_chunks)

    def convert_single_pdf(self, pdf_path: str, save_to: Optional[str] = None) -> str:
        """
        Convert a single PDF file to a JSON string using the chosen LLM strategy.
        If save_to is not None, the JSON is also saved to that path.

        Returns:
            A string containing JSON (or best-effort JSON).
        """
        pdf_content = self.extract_text_from_pdf(pdf_path)

        # Split if needed to avoid hitting max context length
        text_chunks = self._split_into_chunks(pdf_content, max_chars=3000)

        if len(text_chunks) == 1:
            # If the PDF is small enough, do a single LLM call
            prompt_messages = self.strategy.create_prompt_messages(pdf_content)
            response = self.llm.predict_messages(prompt_messages)
            json_result = self.strategy.parse_llm_output(response.content)
        else:
            # Otherwise, convert each chunk separately
            partial_results = self._convert_chunks_to_json_list(text_chunks)
            # Minimal approach: Wrap them into a single JSON structure
            combined = {
                "chunk_count": len(partial_results),
                "chunks": partial_results
            }
            json_result = json.dumps(combined, indent=2)

        if save_to:
            with open(save_to, "w", encoding="utf-8") as f:
                f.write(json_result)

        return json_result

    def convert_directory_of_pdfs(self, directory_path: str, output_dir: str) -> List[str]:
        """
        Convert all PDF files in a directory to JSON strings.
        Returns a list of JSON strings, one per PDF file.

        Also writes each resulting JSON to an output directory (creating it if needed).
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"Not a valid directory: {directory_path}")

        os.makedirs(output_dir, exist_ok=True)

        pdf_files = [
            f for f in os.listdir(directory_path)
            if f.lower().endswith(".pdf")
        ]
        pdf_files.sort()

        json_results = []
        for pdf_file in pdf_files:
            full_path = os.path.join(directory_path, pdf_file)
            # Create an output name
            base_name = os.path.splitext(pdf_file)[0]
            out_path = os.path.join(output_dir, f"{base_name}.json")

            # Convert single pdf
            json_result = self.convert_single_pdf(full_path, save_to=out_path)
            json_results.append(json_result)

        return json_results
