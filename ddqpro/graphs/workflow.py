from langgraph.graph import StateGraph, START, END
from ddqpro.models.state import DDQState
from ddqpro.agents.analyzer import DocumentAnalyzer
from ddqpro.agents.extractor import DocumentExtractor
from ddqpro.agents.reflector import ExtractionReflector
from ddqpro.graphs.response_workflow import ResponseWorkflow


def build_workflow() -> StateGraph:
    """Construct the DDQ processing workflow"""
    workflow = StateGraph(DDQState)
    response_workflow = ResponseWorkflow()

    # Add nodes
    workflow.add_node("analyze", DocumentAnalyzer().analyze)
    workflow.add_node("extract", DocumentExtractor().extract)
    workflow.add_node("reflect", ExtractionReflector().reflect)
    workflow.add_node("process_responses", process_responses)

    # Add edges
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "extract")
    workflow.add_edge("extract", "reflect")

    # Add conditional edges based on reflection
    workflow.add_conditional_edges(
        "reflect",
        ExtractionReflector().should_retry,
        {
            "extract": "extract",
            "process_responses": "process_responses",  # New path after successful extraction
            END: END
        }
    )

    workflow.add_edge("process_responses", END)

    return workflow.compile()


def process_responses(state: DDQState) -> DDQState:
    """Process responses for each extracted question"""
    response_workflow = ResponseWorkflow().build()

    # Initialize response states if not present
    if 'response_states' not in state:
        state['response_states'] = {}
        state['completed_responses'] = {}

    # Process each question
    for question_id, question in state['extraction_results'].content['questions'].items():
        if question_id not in state['completed_responses']:
            # Initialize response state
            response_state = {
                'question': question,
                'context': [],
                'current_response': None,
                'reflections': [],
                'attempts': 0
            }

            # Process question
            final_state = response_workflow.invoke(response_state)

            # Store final response
            if final_state['current_response']:
                state['completed_responses'][question_id] = final_state['current_response']

            # Store state for debugging/analysis
            state['response_states'][question_id] = final_state

    return state