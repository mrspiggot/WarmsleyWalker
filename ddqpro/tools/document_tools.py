from pathlib import Path
import fitz  # PyMuPDF
from typing import Dict, List
import logging
import pdfplumber

logger = logging.getLogger(__name__)


class PDFExtractor:
    def extract(self, file_path: str) -> Dict:
        try:
            with pdfplumber.open(file_path) as pdf:
                pages_text = []
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    pages_text.append(text)
            full_text = "\n\n".join(pages_text)
            ...
            return {
                "raw_content": full_text,
                "format": "text",
                "success": True
            }

        except Exception as e:
            logger.error(f"Error extracting PDF content: {str(e)}")
            return {
                'error': str(e),
                'success': False
            }

class PDFExtractorFitz:
    def extract(self, file_path: str) -> Dict:
        try:
            logger.info(f"Extracting content from PDF: {file_path}")

            # Open the PDF
            doc = fitz.open(file_path)
            text_content = []

            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_content.append(page.get_text())

            # Combine all text
            full_text = "\n\n".join(text_content)

            logger.debug(f"Extracted content preview: {full_text[:200]}...")
            logger.info(f"Total extracted content length: {len(full_text)}")

            if not full_text:
                raise ValueError("No content extracted from PDF")

            doc.close()
            return {
                'raw_content': full_text,
                'format': 'text',
                'success': True
            }
        except Exception as e:
            logger.error(f"Error extracting PDF content: {str(e)}")
            return {
                'error': str(e),
                'success': False
            }


class DocxExtractor:
    def extract(self, file_path: str) -> Dict:
        # Handle .docx files
        # Could convert to PDF first then use PyMuPDF
        pass