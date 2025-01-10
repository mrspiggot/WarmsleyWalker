import os
from pathlib import Path
import pytest

from ddqpro.tools.pdf_json.converter import PDFtoJSONConverter
from ddqpro.tools.pdf_json.strategies import BasicJSONStrategy

def test_convert_single_pdf(tmp_path):
    """
    Demonstration: Convert a single PDF to JSON, storing the result in `tmp_path`.
    Typically used for ephemeral tests, but you could also store somewhere permanent.
    """

    # 1. Figure out the absolute path to the real PDF
    project_root = Path(__file__).parent.parent
    pdf_path = project_root / "data" / "input" / "mw" / "28.03.2024 Factsheet Lumyna - MW TOPS UCITS Fund GBP B (acc).pdf"

    if not pdf_path.is_file():
        pytest.fail(f"Test PDF not found at {pdf_path}. Please verify the path exists.")

    # 2. Create an output JSON path in tmp_path
    output_json = tmp_path / "output.json"

    # 3. Create converter
    converter = PDFtoJSONConverter(
        llm_model_name="gpt-4",
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        temperature=0.0,
        strategy=BasicJSONStrategy()
    )

    # 4. Convert the PDF
    result = converter.convert_single_pdf(str(pdf_path), save_to=str(output_json))

    # 5. Basic checks
    assert len(result) > 0, "Expected some JSON output"
    assert output_json.exists(), "The JSON file should have been created"


def test_convert_directory_of_pdfs():
    """
    Example showing how to store JSON outputs *beside* the PDFs themselves
    rather than using tmp_path.
    """

    # 1. Build absolute path to the directory that holds PDFs
    project_root = Path(__file__).parent.parent
    pdf_dir = project_root / "data" / "input" / "mw"

    # 2. Create a subdirectory for JSON outputs *right alongside* your PDFs
    #    e.g. "mw/converted_json"
    #    (So if your PDFs are in "mw", you will also have "mw/converted_json")
    output_dir = pdf_dir / "converted_json"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Confirm that the PDF directory isn't empty
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        pytest.skip(f"No PDFs found in {pdf_dir}, skipping directory test.")

    # 4. Create the converter
    converter = PDFtoJSONConverter(
        llm_model_name="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        temperature=0.0,
        strategy=BasicJSONStrategy()
    )

    # 5. Convert entire directory
    results = converter.convert_directory_of_pdfs(str(pdf_dir), str(output_dir))

    # 6. Basic checks
    assert len(results) > 0, "Expected JSON results for each PDF"
    # Make sure at least one JSON file was created
    created_jsons = list(output_dir.glob("*.json"))
    assert created_jsons, "No .json files created in the output directory"

    # Optionally: you can print the output file paths for debugging
    print("Created JSON files:")
    for jsonfile in created_jsons:
        print(f"   {jsonfile}")
