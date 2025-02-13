from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ddqpro.models.state import DDQState, Answer, Question, QuestionMetadata
from ddqpro.rag.retriever import RAGRetriever
from ddqpro.utils.cost_tracking import CostTracker
from ddqpro.models.llm_manager import LLMManager
from ddqpro.agents.enhanced_processor import EnhancedProcessor
from ddqpro.rag.retriever import RAGRetriever
from ddqpro.rag.document_processor import CorpusProcessor

logger = logging.getLogger(__name__)


def sync_process_questions(state: DDQState) -> DDQState:
    """Synchronous wrapper for async question processing"""
    logger.info("Starting sync_process_questions with detailed state inspection")

    # Instantiate CorpusProcessor and ingest documents
    corpus_processor = CorpusProcessor(corpus_dir="ddqpro/gui/data/corpus")
    corpus_processor.ingest_documents(reset_db=False)

    # Create a RAGRetriever using the pre-populated vector store
    retriever = RAGRetriever(vector_store=corpus_processor.vector_store)
    processor = EnhancedProcessor(retriever=retriever)

    logger.debug(f"State keys: {state.keys()}")

    if not state.get('json_output'):
        logger.error("No json_output in state")
        return state

    sections = state['json_output'].get('sections', [])
    if not sections:
        logger.error("No sections found in json_output")
        return state

    questions_list = []
    for section in sections:
        for q_dict in section.get('questions', []):
            try:
                # Skip questions with incomplete data
                if not all(key in q_dict for key in ['id', 'text', 'type', 'metadata']):
                    logger.warning(f"Skipping question {q_dict.get('id', 'unknown')}: incomplete data")
                    continue

                # Create metadata with defaults for missing fields
                metadata = QuestionMetadata(
                    category=q_dict.get('metadata', {}).get('category', 'unknown'),
                    subcategory=q_dict.get('metadata', {}).get('subcategory', 'general'),
                    context=q_dict.get('metadata', {}).get('context', section['title'].lower())
                )

                # Create question object
                question = Question(
                    id=q_dict['id'],
                    text=q_dict['text'],
                    type=q_dict.get('type', 'text'),
                    required=q_dict.get('required', True),
                    metadata=metadata,
                    section=section['title']
                )
                questions_list.append(question)
                logger.debug(f"Created Question object for {q_dict['id']}")
            except Exception as e:
                logger.error(f"Error converting question {q_dict.get('id', 'unknown')} to Question object: {str(e)}",
                             exc_info=True)
                continue

    if not questions_list:
        logger.error("No valid questions created")
        return state

    try:
        logger.info(f"Processing {len(questions_list)} questions")
        answers = asyncio.run(processor.process_questions(
            questions=questions_list,
            section=sections[0]['title'] if sections else "default"
        ))

        if answers:
            if 'json_output' not in state:
                state['json_output'] = {}

            state['json_output']['answers'] = {
                qid: {
                    "text": answer.text,
                    "confidence": answer.confidence,
                    "sources": [
                        {"file": src.get('source', 'Unknown'),
                         "excerpt": src.get('content', '')[:200]}
                        for src in (answer.sources or [])
                    ],
                    "metadata": answer.metadata
                }
                for qid, answer in answers.items()
            }
            logger.info(f"Successfully processed {len(answers)} answers")

    except Exception as e:
        logger.error(f"Error processing questions: {str(e)}", exc_info=True)

    return state


class SectionProcessor:
    """Processes DDQ sections in parallel"""

    def __init__(self):
        self.retriever = RAGRetriever()
        self.llm_manager = LLMManager()
        self.context_cache = {}  # Section-level context cache
        self.cost_tracker = CostTracker()

    async def process_section(self, section_name: str, questions: List[Question],
                              batch_size: int = 10) -> Dict[str, Answer]:
        """Process all questions in a section"""
        logger.info(f"Processing section {section_name} with {len(questions)} questions")

        # Get section-level context once
        section_context = await self.retriever.aget_section_context(section_name)

        # Track embedding costs
        self.cost_tracker.track_call(
            model="text-embedding-3-small",
            tokens_in=len(section_context) * 500,  # Approximate tokens per chunk
            tokens_out=0,
            endpoint="embedding"
        )
        self.context_cache[section_name] = section_context

        # Split questions into batches
        batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]

        # Process batches concurrently
        answers = {}
        for batch in batches:
            batch_answers = await self.process_batch(section_name, batch)
            answers.update(batch_answers)

        return answers

    async def process_batch(self,
                            section_name: str,
                            questions: List[Question]) -> Dict[str, Answer]:
        """Process a batch of questions concurrently"""
        tasks = []
        for question in questions:
            task = self.process_question(section_name, question)
            tasks.append(task)

        # Run all questions in batch concurrently
        batch_results = await asyncio.gather(*tasks)

        # Combine results
        return {q.id: a for q, a in zip(questions, batch_results)}

    async def process_questions(self, questions: List[Question], section: str) -> Dict[str, Answer]:
        """Process questions with optimized strategies and detailed logging"""
        logger.info(f"Processing {len(questions)} questions from section {section}")

        # Debug state content
        print("\n=== Available Documents ===")
        await self.retriever.debug_print_documents()

        try:
            # Analyze questions
            analyzed_questions = [
                {
                    'question': q,
                    'analysis': self.text_analyzer.analyze_question_type(q.text)
                }
                for q in questions
            ]
            logger.debug(f"Question analysis complete: {len(analyzed_questions)} questions analyzed")

            # Group questions
            grouped_questions = self._group_questions(analyzed_questions)
            logger.debug(f"Questions grouped into {len(grouped_questions)} groups")

            # Process each group
            results = {}
            for group_type, group_data in grouped_questions.items():
                logger.info(f"Processing group {group_type} with {len(group_data['questions'])} questions")

                # Get context for each question in the group
                for question in group_data['questions']:
                    context = await self.retriever.aget_relevant_context(
                        question=question.text,
                        metadata_filter={'doc_type': 'DDQ'}
                    )
                    print(f"\nRetrieved context for question {question.id}:")
                    print(f"Context length: {len(context)}")
                    if context:
                        print(f"First context item: {context[0][:200]}...")
                    else:
                        print("No context found!")

                q_type = group_data['type']
                strategy = self.strategies.get(q_type, self.strategies['factual'])
                logger.debug(f"Using strategy: {q_type}")

                try:
                    group_results = await self._process_group(
                        questions=group_data['questions'],
                        strategy=strategy,
                        shared_context=group_data.get('shared_context')
                    )
                    logger.debug(f"Group {group_type} processing complete, got {len(group_results)} answers")
                    results.update(group_results)
                except Exception as e:
                    logger.error(f"Error processing group {group_type}: {str(e)}", exc_info=True)

            logger.info(f"Successfully processed {len(results)} questions")
            logger.debug(f"Answer IDs: {list(results.keys())}")
            return results

        except Exception as e:
            logger.error(f"Error in process_questions: {str(e)}", exc_info=True)
            raise

    def _needs_additional_context(self, question: Question) -> bool:
        """Determine if question needs context beyond section-level"""
        # Simple questions about basic info might not need additional context
        if question.metadata.category in ['company', 'contact']:
            return False
        return True

    async def _generate_answer(self, question: Question, context: List[Dict]) -> str:
        """Generate answer using LLM"""
        prompt = ChatPromptTemplate.from_template("""
                Use your skill and judgement to give the most likely answer to this question given the Context. 
                The answer may not be explicit but use your judgement to give the most likely answer.
                Question: {question}
                Category: {category}
                Context: {context}

                Answer the question asked in the context of an employee of an investment management firm. Bear in mind the 
                context and vernacular of the fund management industry. Things like 'Company name' and 'Manager name' or 
                'Investment Manager' all mean the same thing. Think holistically about this for all questions and pull out the 
                information based on semantics not matching of keywords
                
                Always provide an answer, provide the best answer you can given the Context

                Answer:
                """)

        chain = prompt | self.llm_manager.llm
        response = await chain.ainvoke({
            "question": question.text,
            "category": question.metadata.category,
            "context": "\n\n".join([doc['content'] for doc in context])
        })

        # Track completion costs
        self.cost_tracker.track_call(
            model="gpt-4-turbo-preview",
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            endpoint="completion"
        )

        return response