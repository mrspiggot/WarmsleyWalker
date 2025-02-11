##############################################################################
# Main Converter Class
##############################################################################
import os
import json
from typing import List, Union, Optional
from ddqpro.tools.pdf_json.strategies import PDFExtractionStrategy, BasicJSONStrategy


import pdfplumber
from langchain.chat_models import ChatOpenAI  # or your chosen LLM library
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
            llm_model_name: str = "gpt-4o",  # default you wanted
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
        Returns:
            A string containing JSON (or best-effort JSON)
            that represents the content of the PDF.
        If save_to is not None, the JSON is also saved to that path.
        """
        pdf_content = self.extract_text_from_pdf(pdf_path)

        # Strategy-based prompt creation
        prompt_messages = self.strategy.create_prompt_messages(pdf_content)

        # We'll call LLM with our prompt messages
        response = self.llm.predict_messages(prompt_messages)

        # Strategy-based parse
        json_result = self.strategy.parse_llm_output(response.content)

        # Optionally save to disk
        if save_to:
            with open(save_to, "w", encoding="utf-8") as f:
                f.write(json_result)

        return json_result

    def convert_directory_of_pdfs(self, directory_path: str, output_dir: str) -> List[str]:
        """
        Convert all PDF files in a directory to JSON strings.
        Returns:
            A list of JSON strings, one per PDF file (in alphabetical order).
        Also writes each resulting JSON to an output directory
        (creating it if needed).
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