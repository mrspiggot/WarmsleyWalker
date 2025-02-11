# ddqpro/tools/document_tools_ocr.py

import os
import logging
from typing import Dict
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path, pdf2image
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class OCRConfig:
    """
    Configuration for the OCR extractor.
    """
    dpi: int = 300
    poppler_path: str = field(default_factory=str)  # If needed on Windows or custom location
    lang: str = "eng"  # Tesseract language code (e.g., "eng", "fra")
    # Add more config fields if you want, such as page range, grayscale, etc.


class PDFExtractorOCR:
    """
    Extracts text via OCR by converting each PDF page to an image and then applying Tesseract.
    """

    def __init__(self, config: OCRConfig = OCRConfig()):
        self.config = config

    def extract(self, file_path: str) -> Dict:
        """
        Perform OCR on the given PDF.

        Returns:
            A dict:
            {
                "raw_content": "Text content from all pages combined",
                "format": "text",
                "success": True/False,
                "error": <error-message-if-any>
            }
        """
        results = {
            "raw_content": "",
            "format": "text",
            "success": False,
            "error": ""
        }

        try:
            pdf_file = Path(file_path)
            if not pdf_file.is_file():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Convert PDF to image(s)
            logger.info(f"OCR Extractor: Converting PDF to images for {pdf_file}")
            images = convert_from_path(
                pdf_file,
                dpi=self.config.dpi,
                poppler_path=self.config.poppler_path or None
            )

            logger.info(f"OCR Extractor: {len(images)} page(s) detected for {pdf_file}")

            # OCR each page
            all_text = []
            for idx, img in enumerate(images):
                # You can add your own pre-processing steps here if needed
                logger.debug(f"OCR Extractor: Starting OCR on page {idx+1} of {len(images)}")

                page_text = pytesseract.image_to_string(
                    img,
                    lang=self.config.lang
                )
                all_text.append(page_text)

            # Join them into one big chunk
            combined_text = "\n\n".join(all_text)
            logger.info(f"OCR Extractor: Extracted {len(combined_text)} characters from {pdf_file}")

            results["raw_content"] = combined_text
            results["success"] = True

        except Exception as e:
            msg = str(e)
            logger.error(f"OCR Extractor error: {msg}", exc_info=True)
            results["error"] = msg

        return results
