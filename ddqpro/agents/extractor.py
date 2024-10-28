from typing import Dict, List
import logging
from datetime import datetime
from ddqpro.models.state import DDQState, Answer
from ddqpro.rag.retriever import RAGRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

ANSWER_TEMPLATE = """You are an expert at answering Due Diligence Questionnaire (DDQ) questions.
Use the provided context to generate a detailed, accurate answer to the question.

Question: {question}
Category: {category}
Context: {context}

Guidelines:
- Be specific and precise
- Include relevant facts and figures from the context
- If the context doesn't provide enough information, state "Based on the available information, I cannot provide a specific answer."
- Format appropriately for the question type
- Be professional and concise

Answer:"""


class DocumentExtractor:
    def __init__(self):
        self.retriever = RAGRetriever()
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
        self.answer_prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

    def extract(self, state: DDQState) -> DDQState:
        """Extract content and generate answers"""
        logger.info(f"Extracting content from: {state['input_path']}")

        if not state.get('extraction_results'):
            logger.warning("No extraction results found in state")
            return state

        try:
            # Generate answers for each question
            answers = {}
            extraction_results = state['extraction_results']

            for question_id, question in extraction_results.content['questions'].items():
                logger.info(f"Processing question {question_id}: {question['text'][:50]}...")

                try:
                    # Retrieve relevant context
                    context = self.retriever.get_relevant_context(
                        question=question['text'],
                        metadata_filter={'doc_type': 'DDQ'}
                    )

                    logger.debug(f"Retrieved {len(context)} context chunks for question {question_id}")

                    # Generate answer
                    chain = self.answer_prompt | self.llm
                    response = chain.invoke({
                        "question": question['text'],
                        "category": question['metadata']['category'],
                        "context": "\n\n".join([doc['content'] for doc in context])
                    })

                    logger.debug(f"Generated answer for question {question_id}")

                    # Create answer object
                    answer = Answer(
                        text=response.content,
                        confidence=0.85,
                        sources=context,
                        metadata={
                            "question_type": question['type'],
                            "generated_at": datetime.now().isoformat()
                        }
                    )

                    answers[question_id] = answer
                    logger.info(f"Successfully generated answer for question {question_id}")

                except Exception as e:
                    logger.error(f"Error processing question {question_id}: {str(e)}")
                    continue

            # Update state with answers
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

            logger.info(f"Generated {len(answers)} answers")
            return state

        except Exception as e:
            logger.error(f"Error in extraction process: {str(e)}")
            raise