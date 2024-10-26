from langgraph.graph import StateGraph, START, END
from ddqpro.models.state import DDQState
from ddqpro.agents.analyzer import DocumentAnalyzer
from ddqpro.agents.extractor import DocumentExtractor
from ddqpro.agents.reflector import ExtractionReflector

def build_workflow() -> StateGraph:
    """Construct the DDQ processing workflow"""
    workflow = StateGraph(DDQState)

    # Add nodes
    workflow.add_node("analyze", DocumentAnalyzer().analyze)
    workflow.add_node("extract", DocumentExtractor().extract)
    workflow.add_node("reflect", ExtractionReflector().reflect)

    # Add edges
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "extract")
    workflow.add_edge("extract", "reflect")  # Add this edge

    # Add conditional edges based on reflection
    workflow.add_conditional_edges(
        "reflect",
        ExtractionReflector().should_retry,
        {
            "extract": "extract",  # Changed from "Try different extraction" to match node name
            END: END               # Changed from "Extraction complete" to END
        }
    )

    return workflow.compile()