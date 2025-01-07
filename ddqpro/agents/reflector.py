from ddqpro.models.state import DDQState
from langgraph.graph import END
import logging

logger = logging.getLogger(__name__)

# In agents/reflector.py

class ExtractionReflector:
    def reflect(self, state: DDQState) -> DDQState:
        """Reflect on the quality of extraction and suggest improvements"""
        logger.info(f"Reflecting on extraction from: {state['input_path']}")
        return state

    def should_retry(self, state: DDQState) -> str:
        """Decide whether to retry extraction or end processing"""
        return END  # Always end for now