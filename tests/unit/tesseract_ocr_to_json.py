# script_tesseract_ocr_to_json.py

import os
import pytesseract
from pdf2image import convert_from_path
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.docstore.document import Document
from dotenv import load_dotenv

def tesseract_extract_text(pdf_path: str, dpi=300, tesseract_cmd=None) -> str:
    """Extract text from PDF via Tesseract OCR."""
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    images = convert_from_path(pdf_path, dpi=dpi)
    all_text = []
    for img in images:
        page_text = pytesseract.image_to_string(img)
        all_text.append(page_text)
    return "\n".join(all_text)

def main():
    # 1) Extract text from the PDF
    pdf_file = "../../data/input/mw/28.03.2024 Factsheet Lumyna - MW TOPS UCITS Fund GBP B (acc).pdf"
    extracted_text = tesseract_extract_text(pdf_file)

    # 2) Define a structured output parser (to ensure valid JSON)
    response_schemas = [
        ResponseSchema(name="title", description="Document or fund title"),
        ResponseSchema(name="manager", description="Name of manager/company"),
        ResponseSchema(name="key_points", description="List of important bullet points"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    # 3) Prompt template
    template = """
You are given the following text from a PDF:
{pdf_text}

Extract the following information and return it in valid JSON:
{format_instructions}
"""
    prompt = PromptTemplate(
        input_variables=["pdf_text", "format_instructions"],
        template=template
    )

    # 4) Run an LLM chain
    llm = ChatOpenAI(
        model_name="gpt-4o",  # or "gpt-4"
        temperature=0.0,
        openai_api_key=os.environ.get("OPENAI_API_KEY", "")
    )

    chain = LLMChain(
        llm=llm,
        prompt=prompt
    )
    # chain = prompt | llm

    response = chain.run({
        "pdf_text": extracted_text,
        "format_instructions": format_instructions
    })

    # 5) Parse and print JSON
    try:
        json_output = output_parser.parse(response)
        print("\nExtracted JSON:\n", json_output)
    except Exception as e:
        print("Failed to parse JSON:", e)
        print("Raw LLM response:\n", response)

if __name__ == "__main__":
    main()
