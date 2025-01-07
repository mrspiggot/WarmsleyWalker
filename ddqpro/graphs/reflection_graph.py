from typing import Literal, TypedDict, Dict, Optional
from langgraph.graph import StateGraph, START, END
from ddqpro.agents.json_validator import JSONValidator
from ddqpro.models.llm_factory import LLMFactory
import logging

logger = logging.getLogger(__name__)


class ReflectionState(TypedDict):
    content: str
    attempts: int
    current_model: str
    validation_error: Optional[str]
    validated_json: Optional[Dict]


class ReflectionGraph:
    def __init__(self, max_attempts: int = 3):
        self.validator = JSONValidator()
        self.llm_factory = LLMFactory()
        self.max_attempts = max_attempts

    def build(self) -> StateGraph:
        workflow = StateGraph(ReflectionState)

        # Add nodes
        workflow.add_node("validate", self.validate_json)
        workflow.add_node("retry", self.retry_generation)

        # Add edges
        workflow.add_edge(START, "validate")

        # Add conditional edges based on validation
        workflow.add_conditional_edges(
            "validate",
            self.should_continue,
            {
                "retry": "validate",
                END: END
            }
        )

        return workflow.compile()

    def validate_json(self, state: ReflectionState) -> ReflectionState:
        """Validate JSON and update state"""
        is_valid, validated_json, error = self.validator.validate(state["content"])

        return {
            **state,
            "validation_error": error if not is_valid else None,
            "validated_json": validated_json if is_valid else None
        }

    def retry_generation(self, state: ReflectionState) -> ReflectionState:
        """Retry with different model or prompt"""
        attempts = state["attempts"] + 1

        try:
            # Try next available model
            llm = self.llm_factory.get_next_model()

            # Generate new content
            response = llm.invoke(state["original_prompt"])

            return {
                **state,
                "content": response.content,
                "attempts": attempts,
                "current_model": llm.model_name
            }
        except Exception as e:
            logger.error(f"Retry failed: {str(e)}")
            return {
                **state,
                "attempts": attempts,
                "validation_error": str(e)
            }

    def should_continue(self, state: ReflectionState) -> Literal["retry", "end"]:
        """Decide whether to continue trying"""
        # End if we have valid JSON
        if state["validated_json"] is not None:
            return END

        # End if we've exceeded max attempts
        if state["attempts"] >= self.max_attempts:
            return END

        # Otherwise retry
        return "retry"