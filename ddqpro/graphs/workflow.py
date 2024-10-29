from langgraph.graph import StateGraph, START, END
from ddqpro.models.state import DDQState
from ddqpro.agents.analyzer import DocumentAnalyzer
from ddqpro.agents.extractor import DocumentExtractor
from ddqpro.agents.reflector import ExtractionReflector
from ddqpro.graphs.response_workflow import ResponseWorkflow
from ddqpro.agents.enhanced_processor import EnhancedProcessor
from ddqpro.rag.retriever import RAGRetriever
import asyncio
from ddqpro.models.state import Question, QuestionMetadata
import logging

logger = logging.getLogger(__name__)


def sync_process_questions(state: DDQState):
    """Synchronous wrapper for async process_questions"""
    logger.info("Starting sync_process_questions")
    processor = EnhancedProcessor(retriever=RAGRetriever())

    if not state.get('extraction_results'):
        logger.error("No extraction_results in state")
        return state

    questions_dict = state['extraction_results'].content.get('questions', {})
    if not questions_dict:
        logger.error("No questions found in extraction_results")
        return state

    # Log the questions found
    logger.info(f"Found {len(questions_dict)} questions to process")
    for qid, q_dict in questions_dict.items():
        logger.debug(f"Question {qid}: {q_dict.get('text', '')[:100]}...")

    # Convert question dictionaries to Question objects
    questions_list = []
    for qid, q_dict in questions_dict.items():
        try:
            # Create metadata object
            metadata = QuestionMetadata(
                category=q_dict.get('metadata', {}).get('category', 'unknown'),
                subcategory=q_dict.get('metadata', {}).get('subcategory'),
                context=q_dict.get('metadata', {}).get('context')
            )

            # Create question object
            question = Question(
                id=qid,
                text=q_dict['text'],
                type=q_dict.get('type', 'text'),
                required=q_dict.get('required', False),
                metadata=metadata,
                section=q_dict.get('section', 'default')
            )
            questions_list.append(question)
            logger.debug(f"Successfully created Question object for {qid}")
        except Exception as e:
            logger.error(f"Error converting question {qid} to Question object: {str(e)}")
            continue

    if not questions_list:
        logger.error("No valid questions created")
        return state

    section = questions_list[0].section if questions_list else "default"
    logger.info(f"Processing {len(questions_list)} questions from section '{section}'")

    try:
        logger.info("Starting async question processing")
        answers = asyncio.run(processor.process_questions(
            questions=questions_list,
            section=section
        ))

        if not answers:
            logger.error("No answers generated")
            return state

        logger.info(f"Generated {len(answers)} answers")

        # Update state with answers
        if 'json_output' not in state:
            state['json_output'] = {}

        # Log each answer being added
        for qid, answer in answers.items():
            logger.debug(f"Adding answer for question {qid}: {answer.text[:100]}...")

        state['json_output']['answers'] = {
            qid: {
                "text": answer.text,
                "confidence": answer.confidence,
                "sources": [
                    {"file": src['source'], "excerpt": src['content'][:200]}
                    for src in answer.sources
                ],
                "metadata": answer.metadata
            }
            for qid, answer in answers.items()
        }

        logger.info("Successfully updated state with answers")

    except Exception as e:
        logger.error(f"Error processing questions: {str(e)}", exc_info=True)

    return state
def sync_process_questions_old(state: DDQState):
    """Synchronous wrapper for async process_questions"""
    processor = EnhancedProcessor(retriever=RAGRetriever())
    # Get questions from state
    if not state.get('extraction_results'):
        return state

    questions_dict = state['extraction_results'].content.get('questions', {})
    section = list(questions_dict.values())[0]['section'] if questions_dict else "default"

    # Convert question dictionaries to Question objects
    questions_list = []
    for q_dict in questions_dict.values():
        try:
            # Ensure that metadata is properly constructed
            metadata = QuestionMetadata(**q_dict.get('metadata', {}))
            question = Question(
                id=q_dict.get('id'),
                text=q_dict.get('text'),
                type=q_dict.get('type'),
                required=q_dict.get('required', False),
                metadata=metadata,
                section=q_dict.get('section', '')
            )
            questions_list.append(question)
        except Exception as e:
            print(f"Error converting question dict to Question object: {e}")

    # Run async function in sync context
    try:
        answers = asyncio.run(processor.process_questions(
            questions=questions_list,
            section=section
        ))

        # Update state with answers
        if 'json_output' not in state:
            state['json_output'] = {}

        state['json_output']['answers'] = {
            qid: {
                "text": answer.text,
                "confidence": answer.confidence,
                "sources": [
                    {"file": src['source'], "excerpt": src['content'][:200]}
                    for src in answer.sources
                ],
                "metadata": answer.metadata
            }
            for qid, answer in answers.items()
        }

    except Exception as e:
        print(f"Error processing questions: {str(e)}")

    return state


def build_workflow() -> StateGraph:
    """Construct the DDQ processing workflow"""
    workflow = StateGraph(DDQState)

    # Add nodes
    workflow.add_node("analyze", DocumentAnalyzer().analyze)
    workflow.add_node("process", sync_process_questions)  # Using sync wrapper
    workflow.add_node("reflect", ExtractionReflector().reflect)

    # Add edges
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "process")
    workflow.add_edge("process", "reflect")

    # Add conditional edges based on reflection
    workflow.add_conditional_edges(
        "reflect",
        ExtractionReflector().should_retry,
        {
            "process": "process",  # Retry processing if needed
            END: END  # End if processing is complete
        }
    )

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