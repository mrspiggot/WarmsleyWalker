from typing import Dict, Optional, Tuple
import json
from pydantic import ValidationError
from ddqpro.models.json_schema import DDQAnalysis
import logging

logger = logging.getLogger(__name__)


class JSONValidator:
    def validate(self, content: str) -> Tuple[bool, Optional[Dict], str]:
        """Validate JSON content against schema"""
        try:
            # Try to parse as JSON
            if isinstance(content, str):
                data = json.loads(content)
            else:
                data = content

            # Validate against schema
            DDQAnalysis(**data)
            return True, data, ""

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return False, None, f"Invalid JSON format: {str(e)}"

        except ValidationError as e:
            logger.error(f"Schema validation error: {str(e)}")
            return False, None, f"Schema validation failed: {str(e)}"

        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False, None, f"Validation failed: {str(e)}"