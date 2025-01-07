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
from ddqpro.models.llm_manager import LLMManager
from ddqpro.graphs.reflection_graph import ReflectionGraph

logger = logging.getLogger(__name__)


def sync_process_questions(state: DDQState):
    """Synchronous wrapper for async process_questions"""
    logger.info("Starting sync_process_questions with detailed debugging")
    processor = EnhancedProcessor(retriever=RAGRetriever())

    # Debug state content
    logger.debug(f"Initial state keys: {state.keys()}")
    logger.debug(f"Extraction results present: {bool(state.get('extraction_results'))}")

    if not state.get('extraction_results'):
        logger.error("No extraction_results in state")
        return state

    questions_dict = state['extraction_results'].content.get('questions', {})
    logger.debug(f"Number of questions found: {len(questions_dict)}")
    logger.debug(f"Question IDs: {list(questions_dict.keys())}")

    if not questions_dict:
        logger.error("No questions found in extraction_results")
        return state

    # Convert question dictionaries to Question objects with detailed logging
    questions_list = []
    for qid, q_dict in questions_dict.items():
        try:
            logger.debug(f"Processing question {qid}")
            logger.debug(f"Question data: {q_dict}")

            # Create metadata object
            metadata = QuestionMetadata(
                category=q_dict.get('metadata', {}).get('category', 'unknown'),
                subcategory=q_dict.get('metadata', {}).get('subcategory'),
                context=q_dict.get('metadata', {}).get('context')
            )
            logger.debug(f"Created metadata for question {qid}")

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
            logger.error(f"Error converting question {qid} to Question object: {str(e)}", exc_info=True)
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

        logger.debug(f"Answers received: {bool(answers)}")
        if answers:
            logger.debug(f"Number of answers: {len(answers)}")
            logger.debug(f"Answer IDs: {list(answers.keys())}")
        else:
            logger.error("No answers generated")

        # Update state with answers
        if 'json_output' not in state:
            state['json_output'] = {}
            logger.debug("Created new json_output in state")

        if answers:
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
            logger.info(f"Successfully added {len(answers)} answers to state")
            logger.debug(f"Final state keys: {state['json_output'].keys()}")
        else:
            logger.error("No answers to add to state")

    except Exception as e:
        logger.error(f"Error processing questions: {str(e)}", exc_info=True)
        logger.debug("Full state at error:", state)

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

# In graphs/workflow.py

def build_workflow(provider: str = None, model_name: str = None) -> StateGraph:
    """Construct the DDQ processing workflow"""
    logger.info("\nBuilding workflow...")

    # Initialize LLM first, before building workflow
    if provider and model_name:
        logger.info(f"Initializing LLM with provider: {provider}, model: {model_name}")
        llm_manager = LLMManager()
        llm_manager.initialize(provider, model_name)
        logger.info("LLM initialization complete")

    workflow = StateGraph(DDQState)

    # Add nodes
    analyzer = DocumentAnalyzer()

    def analyze_with_debug(state: DDQState) -> DDQState:
        logger.info("\nStarting analysis node")
        try:
            result = analyzer.analyze(state)
            logger.info("Analysis completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in analysis node: {str(e)}")
            raise

    # Define the basic workflow
    workflow.add_node("analyze", analyze_with_debug)
    workflow.add_node("process", sync_process_questions)
    workflow.add_node("reflect", ExtractionReflector().reflect)

    # Define the linear flow
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "process")
    workflow.add_edge("process", "reflect")
    workflow.add_edge("reflect", END)

    logger.info("Workflow build complete")
    return workflow.compile()

def build_workflow_old(provider: str = None, model_name: str = None) -> StateGraph:
    """Construct the DDQ processing workflow"""
    print("\nBuilding workflow...")  # Debug print

    # Initialize LLM first, before building workflow
    if provider and model_name:
        print(f"Initializing LLM with provider: {provider}, model: {model_name}")  # Debug print
        llm_manager = LLMManager()
        llm_manager.initialize(provider, model_name, temperature=0)
        print("LLM initialization complete")  # Debug print
    else:
        print("No provider/model specified, skipping LLM initialization")  # Debug print

    workflow = StateGraph(DDQState)
    # Initialize reflection graph
    reflection = ReflectionGraph(max_attempts=3)
    reflection_graph = reflection.build()

    # Add reflection node
    def analyze_with_reflection(state: DDQState) -> DDQState:
        analyzer = DocumentAnalyzer()

        # Initial analysis attempt
        try:
            result = analyzer.analyze(state)

            # Validate JSON through reflection graph
            reflection_state = {
                "content": result.get("json_output", {}),
                "attempts": 0,
                "current_model": model_name,
                "validation_error": None,
                "validated_json": None,
                "original_prompt": state["input_prompt"]
            }

            final_state = reflection_graph.invoke(reflection_state)

            if final_state["validated_json"]:
                result["json_output"] = final_state["validated_json"]
            else:
                raise ValueError(f"Failed to generate valid JSON: {final_state['validation_error']}")

            return result

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

    workflow.add_node("analyze", analyze_with_reflection)

    # Add nodes with debug prints
    print("Adding workflow nodes...")  # Debug print
    analyzer = DocumentAnalyzer()

    def analyze_with_debug(state: DDQState) -> DDQState:
        print("\nStarting analysis node")  # Debug print
        try:
            result = analyzer.analyze(state)
            print("Analysis completed successfully")  # Debug print
            return result
        except Exception as e:
            print(f"Error in analysis node: {str(e)}")  # Debug print
            raise

    workflow.add_node("analyze", analyze_with_debug)
    workflow.add_node("process", sync_process_questions)
    workflow.add_node("reflect", ExtractionReflector().reflect)

    print("Adding workflow edges...")  # Debug print
    # Add edges
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "process")
    workflow.add_edge("process", "reflect")

    # Add conditional edges based on reflection
    workflow.add_conditional_edges(
        "reflect",
        ExtractionReflector().should_retry,
        {
            "process": "process",
            END: END
        }
    )

    print("Workflow build complete")  # Debug print
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