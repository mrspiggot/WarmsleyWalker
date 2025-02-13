# ddqpro/graphs/workflow.py

import logging
from langgraph.graph import StateGraph, START, END
from ddqpro.models.state import DDQState
from ddqpro.agents.analyzer import DocumentAnalyzer
from ddqpro.agents.reflector import ExtractionReflector
from ddqpro.agents.processor import sync_process_questions

logger = logging.getLogger(__name__)


def build_workflow(provider: str = None, model_name: str = None) -> StateGraph:
    """
    Constructs the DDQ processing workflow using the updated DocumentAnalyzer
    (which now uses our extraction pipeline) integrated into a LangGraph state graph.

    The workflow is defined as follows:
      START -> "analyze" -> "process" -> "reflect" -> END
    """
    logger.info("Building workflow with updated DocumentAnalyzer")

    # Initialize the updated DocumentAnalyzer (which now uses our extraction pipeline)
    analyzer = DocumentAnalyzer()

    def analyze_node(state: DDQState) -> DDQState:
        logger.info("Entering analyze_node")
        try:
            # Call the updated analyzer: it extracts text and runs LLM analysis.
            result = analyzer.analyze(state)
            logger.info("Document analysis completed successfully")
            return result
        except Exception as e:
            logger.error("Error in analyze_node: " + str(e))
            raise e

    def process_node(state: DDQState) -> DDQState:
        logger.info("Entering process_node")
        try:
            # Process questions (existing synchronous function)
            result = sync_process_questions(state)
            logger.info("Question processing completed successfully")
            return result
        except Exception as e:
            logger.error("Error in process_node: " + str(e))
            raise e

    # Use our existing reflector to reflect on extraction quality, etc.
    reflector = ExtractionReflector()

    def reflect_node(state: DDQState) -> DDQState:
        logger.info("Entering reflect_node")
        try:
            result = reflector.reflect(state)
            logger.info("Reflection completed successfully")
            return result
        except Exception as e:
            logger.error("Error in reflect_node: " + str(e))
            raise e

    # Create a new state graph with DDQState as our state schema.
    workflow = StateGraph(DDQState)

    # Add nodes to the workflow.
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("process", process_node)
    workflow.add_node("reflect", reflect_node)

    # Define linear edges: START -> analyze -> process -> reflect -> END
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "process")
    workflow.add_edge("process", "reflect")
    workflow.add_edge("reflect", END)

    logger.info("Workflow build complete")
    return workflow.compile()


if __name__ == "__main__":
    # For testing purposes, create a dummy state.
    dummy_state = {
        "input_path": "../../data/input/sample_ddq.pdf",  # Adjust path as needed
        "file_type": ".pdf",
        "current_extractor": "default",
        "extraction_results": None,
        "reflections": [],
        "json_output": None,
        "response_states": {},
        "completed_responses": {},
        "cost_tracking": {},
        "extraction_metadata": None,
    }

    # Optionally, initialize the LLM here via LLMManager if needed.
    wf = build_workflow(provider="ollama", model_name="llama3.1:70b")
    result_state = wf.invoke(dummy_state)
    print("Final state:", result_state)
