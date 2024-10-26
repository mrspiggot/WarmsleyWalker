from ddqpro.models.state import DDQState
from langgraph.graph import END

class ExtractionReflector:
    def reflect(self, state: DDQState) -> DDQState:
        """Reflect on the quality of extraction and suggest improvements"""
        print(f"Reflecting on extraction from: {state['input_path']}")
        return state

    def should_retry(self, state: DDQState) -> str:
        """Decide whether to retry extraction or end processing"""
        # For now, always end. Later we'll add logic to determine if we need to retry
        return END