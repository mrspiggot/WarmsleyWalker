# ddqpro/extractors/base.py

from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ExtractionResult(BaseModel):
    """Structured result from document extraction attempt"""

    content: str = Field(
        description="Extracted text content from document"
    )
    quality_score: float = Field(
        description="Quality score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    extractor_used: str = Field(
        description="Name/type of extractor used"
    )
    extraction_time: float = Field(
        description="Time taken for extraction in seconds"
    )
    metadata: Dict = Field(
        description="Additional extraction metadata",
        default_factory=dict
    )
    error: Optional[str] = Field(
        description="Error message if extraction failed",
        default=None
    )
    success: bool = Field(
        description="Whether extraction was successful",
        default=True
    )

    def __str__(self) -> str:
        """String representation for logging"""
        return (
            f"ExtractionResult(success={self.success}, "
            f"extractor={self.extractor_used}, "
            f"quality={self.quality_score:.2f}, "
            f"content_length={len(self.content)}, "
            f"error={self.error})"
        )

class BaseExtractor(ABC):
    """Base class for all document extractors"""

    def __init__(self):
        self.name = self.__class__.__name__
        logger.debug(f"Initializing {self.name}")

    @abstractmethod
    def extract(self, file_path: str) -> ExtractionResult:
        """
        Extract text content from document.
        """
        pass

    @abstractmethod
    def evaluate_quality(self, text: str) -> float:
        """
        Evaluate quality of extracted text.
        """
        pass

    def _validate_file(self, file_path: str) -> Path:
        """Validate file exists and is supported format"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        return path

    @staticmethod
    def _measure_extraction_time() -> callable:
        """Decorator to measure extraction time."""
        from functools import wraps
        import time

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                if isinstance(result, ExtractionResult):
                    result.extraction_time = elapsed_time
                return result
            return wrapper

        return decorator

class ExtractionError(Exception):
    """Base class for extraction errors"""
    pass

class QualityError(ExtractionError):
    """Raised when extraction quality is below threshold"""
    pass

class UnsupportedFormatError(ExtractionError):
    """Raised when file format is not supported"""
    pass
