import os
from pathlib import Path
import pytest

from ddqpro.tools.pdf_json.converter import PDFtoJSONConverter
from ddqpro.tools.pdf_json.strategies import BasicJSONStrategy

def test_convert_single_pdf(tmp_path):
    """
    Test that references the real PDF in data/input/mw but uses an absolute path
    so we don't rely on the working directory.
    """

    # 1. Build an absolute path to your PDF from the test file's location:
    #    This uses __file__ to find the tests folder, then goes up 1 or 2 levels
    #    until we reach project root, then "data/input/mw".
    project_root = Path(__file__).parent.parent  # Adjust how many .parent you need
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

    permanent_dir = Path("pdf_json_test_artifacts")
    permanent_dir.mkdir(exist_ok=True)

    # Copy the file
    with open(output_json, "rb") as src, open(permanent_dir / "output_copy.json", "wb") as dst:
        dst.write(src.read())


def test_convert_directory_of_pdfs(tmp_path):
    # Setup: pick a directory with 2–3 small PDFs
    pdf_dir = "data/input/mw"
    output_dir = tmp_path / "pdf_json_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = PDFtoJSONConverter(
        llm_model_name="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        temperature=0.0
    )

    results = converter.convert_directory_of_pdfs(str(pdf_dir), str(output_dir))
    assert len(results) > 0, "Expected JSON results for each PDF"
    # Optionally check that each .json file was indeed created
