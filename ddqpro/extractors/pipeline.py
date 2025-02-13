# ddqpro/extractors/pipeline.py

from typing import List
import logging
from .base import ExtractionResult, BaseExtractor

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """
    ExtractionPipeline manages multiple extractors (e.g., direct PDF extraction and OCR extraction)
    and returns the result from the extractor that yields the highest quality output.

    It tries each extractor in order until one produces a quality score above a defined threshold.
    """

    def __init__(self, extractors: List[BaseExtractor], quality_threshold: float = 0.8):
        """
        Args:
            extractors: List of extractor instances to try in order.
            quality_threshold: Quality score above which the result is accepted.
        """
        self.extractors = extractors
        self.quality_threshold = quality_threshold

    def extract(self, file_path: str) -> ExtractionResult:
        """
        Executes the extraction pipeline by trying each extractor in sequence.
        Returns the ExtractionResult with the best quality score.

        Args:
            file_path: Path to the PDF document to extract.

        Returns:
            ExtractionResult: The result from the extractor with the best (or acceptable) quality.
        """
        best_result = None
        best_quality = 0.0

        for extractor in self.extractors:
            logger.info(f"Trying extractor: {extractor.name} for file: {file_path}")
            result = extractor.extract(file_path)
            if result.success:
                quality = extractor.evaluate_quality(result.content)
                logger.info(f"Extractor {extractor.name} returned quality score: {quality:.2f}")
                if quality > best_quality:
                    best_result = result
                    best_quality = quality
                    # If quality exceeds threshold, accept and break
                    if quality >= self.quality_threshold:
                        logger.info(f"Quality threshold met with extractor {extractor.name}")
                        break
            else:
                logger.warning(f"Extractor {extractor.name} failed with error: {result.error}")

        if best_result is None:
            logger.error("No extractor succeeded for file: " + file_path)
            # Return a failed result if none succeeded
            return ExtractionResult(
                content="",
                quality_score=0.0,
                extractor_used="None",
                extraction_time=0.0,
                metadata={"file_path": file_path},
                success=False,
                error="All extractors failed."
            )

        logger.info(f"Best extraction result from {best_result.extractor_used} with quality score: {best_quality:.2f}")
        return best_result
