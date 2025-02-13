# ddqpro/extractors/pdf_extractor.py

import pdfplumber
from typing import Optional, Dict
import logging
import re
from .base import BaseExtractor, ExtractionResult, UnsupportedFormatError

logger = logging.getLogger(__name__)


class PDFDirectExtractor(BaseExtractor):
    """Extract text directly from PDF files using pdfplumber"""

    def __init__(self, quality_threshold: float = 0.5):
        super().__init__()
        self.quality_threshold = quality_threshold
        logger.debug(f"{self.name} initialized with quality threshold {quality_threshold}")

    @BaseExtractor._measure_extraction_time()
    def extract(self, file_path: str) -> ExtractionResult:
        """
        Extract text from PDF using direct extraction
        """
        try:
            path = self._validate_file(file_path)
            if path.suffix.lower() != '.pdf':
                raise UnsupportedFormatError(f"File must be PDF, got: {path.suffix}")

            logger.info(f"Extracting text from PDF: {file_path}")

            with pdfplumber.open(file_path) as pdf:
                pages_text = []
                total_pages = len(pdf.pages)

                # Extract text from each page
                for i, page in enumerate(pdf.pages):
                    logger.debug(f"Processing page {i + 1}/{total_pages}")
                    text = page.extract_text() or ""
                    pages_text.append(text)

                full_text = "\n\n".join(pages_text)

                # Evaluate quality
                quality_score = self.evaluate_quality(full_text)
                logger.info(f"Extraction complete. Quality score: {quality_score:.2f}")

                # Create result object
                return ExtractionResult(
                    content=full_text,
                    quality_score=quality_score,
                    extractor_used=self.name,
                    extraction_time=0.0,  # Will be set by decorator
                    metadata={
                        'num_pages': total_pages,
                        'bytes_extracted': len(full_text),
                        'quality_threshold': self.quality_threshold
                    }
                )

        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            return ExtractionResult(
                content="",
                quality_score=0.0,
                extractor_used=self.name,
                error=str(e),
                success=False,
                extraction_time=0.0,
                metadata={'file_path': file_path}
            )

    def evaluate_quality(self, text: str) -> float:
        """
        Evaluate quality of extracted text based on multiple heuristics
        Returns a score between 0 and 1
        """
        if not text:
            return 0.0

        metrics = {
            'length_score': self._evaluate_length(text),
            'density_score': self._evaluate_character_density(text),
            'structure_score': self._evaluate_structure(text),
            'content_score': self._evaluate_content(text)
        }

        # Weighted average of metrics
        weights = {
            'length_score': 0.3,
            'density_score': 0.3,
            'structure_score': 0.2,
            'content_score': 0.2
        }

        total_score = sum(score * weights[metric] for metric, score in metrics.items())
        logger.debug(f"Quality metrics: {metrics}")

        return min(max(total_score, 0.0), 1.0)  # Clamp between 0 and 1

    def _evaluate_length(self, text: str) -> float:
        """Score based on text length - longer text usually indicates better extraction"""
        min_length = 100  # Minimum expected length
        target_length = 1000  # Length that would give max score

        length = len(text)
        if length < min_length:
            return 0.0
        elif length >= target_length:
            return 1.0
        else:
            return (length - min_length) / (target_length - min_length)

    def _evaluate_character_density(self, text: str) -> float:
        """Score based on ratio of alphanumeric characters to total length"""
        if not text:
            return 0.0

        alphanumeric = sum(c.isalnum() for c in text)
        total = len(text)

        ratio = alphanumeric / total
        # Expect 30-70% density for normal text
        if ratio < 0.3:
            return ratio / 0.3
        elif ratio > 0.7:
            return 1.0 - ((ratio - 0.7) / 0.3)
        else:
            return 1.0

    def _evaluate_structure(self, text: str) -> float:
        """Score based on presence of expected document structure"""
        indicators = [
            '\n',  # Line breaks
            '    ',  # Indentation
            r'\d',  # Numbers
            r'[A-Z]',  # Capital letters
            r'[.,?!]'  # Punctuation
        ]

        scores = []
        for pattern in indicators:
            matches = len(re.findall(pattern, text))
            score = min(matches / 10, 1.0)  # Cap at 1.0
            scores.append(score)

        return sum(scores) / len(scores)

    def _evaluate_content(self, text: str) -> float:
        """Score based on meaningful content indicators"""
        # Look for common document elements
        indicators = {
            'paragraphs': r'\n\n',
            'sentences': r'[.!?]\s',
            'words': r'\b\w+\b'
        }

        scores = []
        for name, pattern in indicators.items():
            matches = len(re.findall(pattern, text))
            if name == 'words':
                score = min(matches / 100, 1.0)  # Expect at least 100 words
            elif name == 'sentences':
                score = min(matches / 10, 1.0)  # Expect at least 10 sentences
            else:
                score = min(matches / 5, 1.0)  # Expect at least 5 paragraphs
            scores.append(score)

        return sum(scores) / len(scores)