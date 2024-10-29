
import asyncio
from typing import List, Dict, Tuple, Union, Any
import logging
from dataclasses import dataclass
from ddqpro.utils.text_utils import TextAnalyzer
from ddqpro.models.state import Question, Answer
from ddqpro.rag.retriever import RAGRetriever
from datetime import datetime
from ddqpro.utils.confidence_scoring import ConfidenceScorer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import time

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStrategy:
    batch_size: int
    concurrent_tasks: int
    context_sharing: bool
    max_retries: int
    timeout: float


class EnhancedProcessor:
    def __init__(self, retriever: RAGRetriever):
        self.retriever = retriever
        self.text_analyzer = TextAnalyzer()
        self.context_cache = {}
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
        self.confidence_scorer = ConfidenceScorer()
        self.previous_answers = {}
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at answering Due Diligence Questionnaire (DDQ) questions.
            Use the provided context to generate detailed, accurate answers. Be specific and professional.
            If the context doesn't provide enough information for a complete answer, acknowledge what's missing.

            Format your answer appropriate to the question type (text, multiple choice, etc).
            If numerical or factual information is available, always include it.
            """),
            ("human", """Question: {question}
            Question Type: {question_type}
            Category: {category}
            Context: {context}

            Please provide a detailed, professional answer:""")
        ])

        # Processing strategies based on question type
        self.strategies = {
            'quantitative': ProcessingStrategy(
                batch_size=5,
                concurrent_tasks=2,
                context_sharing=False,
                max_retries=2,
                timeout=30.0
            ),
            'explanation': ProcessingStrategy(
                batch_size=3,
                concurrent_tasks=2,
                context_sharing=True,
                max_retries=3,
                timeout=45.0
            ),
            'enumeration': ProcessingStrategy(
                batch_size=8,
                concurrent_tasks=3,
                context_sharing=True,
                max_retries=2,
                timeout=25.0
            ),
            'binary': ProcessingStrategy(
                batch_size=15,
                concurrent_tasks=5,
                context_sharing=True,
                max_retries=1,
                timeout=15.0
            ),
            'factual': ProcessingStrategy(
                batch_size=10,
                concurrent_tasks=4,
                context_sharing=True,
                max_retries=1,
                timeout=20.0
            )
        }

    async def process_questions(self,
                                questions: List[Question],
                                section: str) -> Dict[str, Answer]:
        """Process questions with optimized strategies"""
        logger.info(f"Starting to process {len(questions)} questions from section {section}")

        # Analyze questions
        analyzed_questions = [
            {
                'question': q,
                'analysis': self.text_analyzer.analyze_question_type(q.text)
            }
            for q in questions
        ]

        try:
            # Group questions by type and similarity
            grouped_questions = self._group_questions(analyzed_questions)

            # Process each group with appropriate strategy
            results = {}
            for group_type, group_data in grouped_questions.items():
                logger.info(f"Processing group {group_type} with {len(group_data['questions'])} questions")

                q_type = group_data['type']
                strategy = self.strategies.get(q_type, self.strategies['factual'])

                group_results = await self._process_group(
                    questions=group_data['questions'],
                    strategy=strategy,
                    shared_context=group_data.get('shared_context')
                )
                results.update(group_results)

            logger.info(f"Successfully processed {len(results)} questions")
            return results

        except Exception as e:
            logger.error(f"Error in process_questions: {str(e)}", exc_info=True)
            raise

    def _group_questions(self,
                         analyzed_questions: List[Dict]) -> Dict[str, Dict]:
        """Group questions by type and similarity"""
        logger.info("Starting to group questions")

        # First group by type
        type_groups = {}
        for aq in analyzed_questions:
            q_type = aq['analysis']['type']
            if q_type not in type_groups:
                type_groups[q_type] = []
            type_groups[q_type].append(aq['question'])

        logger.info(f"Grouped questions by types: {list(type_groups.keys())}")

        # Then group similar questions within each type
        final_groups = {}
        for q_type, questions in type_groups.items():
            strategy = self.strategies.get(q_type)
            if not strategy:
                logger.warning(f"No strategy found for type {q_type}, using 'factual' strategy")
                strategy = self.strategies['factual']

            if strategy.context_sharing:
                # Group similar questions
                try:
                    similarity_groups = self.text_analyzer.group_similar_questions(
                        questions=questions,
                        threshold=0.6
                    )

                    # Create final groups
                    for group_id, group_data in similarity_groups.items():
                        key = f"{q_type}_{group_id}"
                        final_groups[key] = {
                            'questions': group_data['questions'],
                            'type': q_type,
                            'shared_context': None
                        }
                except Exception as e:
                    logger.error(f"Error processing similarity group for {q_type}: {str(e)}")
                    # Fallback: treat all questions in this type as one group
                    key = f"{q_type}_0"
                    final_groups[key] = {
                        'questions': questions,
                        'type': q_type,
                        'shared_context': None
                    }
            else:
                # Keep questions separate
                key = f"{q_type}_0"
                final_groups[key] = {
                    'questions': questions,
                    'type': q_type,
                    'shared_context': None
                }

        logger.info(f"Final grouping created with {len(final_groups)} groups")
        return final_groups

    async def _process_group(self,
                             questions: List[Question],
                             strategy: ProcessingStrategy,
                             shared_context: Dict = None) -> Dict[str, Answer]:
        """Process a group of questions using the specified strategy"""
        # Split into batches
        batches = [
            questions[i:i + strategy.batch_size]
            for i in range(0, len(questions), strategy.batch_size)
        ]

        results = {}
        async with asyncio.TaskGroup() as group:
            tasks = []
            for batch in batches:
                task = group.create_task(
                    self._process_batch(
                        questions=batch,
                        strategy=strategy,
                        shared_context=shared_context
                    )
                )
                tasks.append(task)

            # Collect results
            for task in tasks:
                batch_results = await task
                results.update(batch_results)

        return results

    async def _process_batch(self,
                             questions: List[Question],
                             strategy: ProcessingStrategy,
                             shared_context: Dict = None) -> Dict[str, Answer]:
        """Process a batch of questions with retries and timeouts"""
        results = {}
        sem = asyncio.Semaphore(strategy.concurrent_tasks)

        async def process_with_retries(question: Question) -> Tuple[str, Answer]:
            async with sem:  # Control concurrency
                for attempt in range(strategy.max_retries):
                    try:
                        # Get context - either shared or specific
                        context = shared_context if shared_context else \
                            await self._get_question_context(question)

                        # Process with timeout
                        async with asyncio.timeout(strategy.timeout):
                            answer = await self._generate_answer(question, context)
                            logger.debug(f"Generated answer for question {question.id}")
                            return question.id, answer

                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Timeout processing question {question.id} "
                            f"(attempt {attempt + 1}/{strategy.max_retries})"
                        )
                        if attempt == strategy.max_retries - 1:
                            raise

                    except Exception as e:
                        logger.error(
                            f"Error processing question {question.id}: {str(e)} "
                            f"(attempt {attempt + 1}/{strategy.max_retries})"
                        )
                        if attempt == strategy.max_retries - 1:
                            raise

                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)

        # Process all questions in batch
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(process_with_retries(question))
                for question in questions
            ]

            for task in tasks:
                try:
                    question_id, answer = await task
                    results[question_id] = answer
                except Exception as e:
                    logger.error(f"Failed to process question: {str(e)}")

        return results

    async def _get_question_context(self, question: Question) -> Dict:
        """Get context specific to a question"""
        # Extract key terms
        terms = self.text_analyzer.extract_key_terms(question.text)
        query = self.text_analyzer.create_optimized_query(terms)

        # Get relevant context
        context = await self.retriever.aget_relevant_context(
            question=query,
            metadata_filter={'doc_type': 'DDQ'}
        )

        return {
            'content': context,
            'query': query,
            'terms': terms
        }

    async def _get_shared_context(self, group_data: Dict) -> Dict:
        """Get or create shared context for a group"""
        # Create cache key from question IDs
        cache_key = "_".join(sorted(q.id for q in group_data['questions']))

        if cache_key in self.context_cache:
            logger.debug(f"Using cached context for group {cache_key}")
            return self.context_cache[cache_key]

        # Extract common terms from all questions
        all_terms = []
        for question in group_data['questions']:
            terms = self.text_analyzer.extract_key_terms(question.text)
            all_terms.extend(terms)

        # Combine terms and create query
        combined_query = self.text_analyzer.create_optimized_query(all_terms)

        # Get context
        context = await self.retriever.aget_relevant_context(
            question=combined_query,
            metadata_filter={'doc_type': 'DDQ'}
        )

        # Cache the results
        self.context_cache[cache_key] = {
            'content': context,
            'query': combined_query,
            'terms': all_terms
        }

        return self.context_cache[cache_key]

    async def _generate_answer(self, question: Question, context: Dict) -> Answer:
        """Generate answer for a question using context"""
        try:
            logger.info(f"Generating answer for question {question.id}")

            # Log the context being used
            logger.debug(f"Using context with {len(context['content'])} chunks")

            # Generate answer
            chain = self.answer_prompt | self.llm
            logger.debug("Invoking LLM chain")
            response = await chain.ainvoke({
                "question": question.text,
                "question_type": question.type,
                "category": question.metadata.category,
                "context": self._format_context(context['content'])
            })

            # Extract the actual response text
            response_text = response.content if hasattr(response, 'content') else str(response)
            logger.debug(f"Generated response of length {len(response_text)}")

            try:
                # Calculate confidence scores
                logger.debug("Calculating confidence scores")
                confidence_score, metrics = self.confidence_scorer.calculate_confidence(
                    question=question.text,
                    answer=response_text,
                    context=context['content'],
                    response_metadata={
                        'token_usage': getattr(response, 'usage', {}).get('total_tokens', 0),
                    },
                    previous_answers=self.previous_answers
                )
                logger.debug(f"Calculated confidence score: {confidence_score}")
            except Exception as e:
                logger.error(f"Error calculating confidence: {str(e)}")
                confidence_score = 0.7
                metrics = None

            # Create answer object
            try:
                logger.debug("Creating Answer object")
                answer = Answer(
                    text=response_text,
                    confidence=confidence_score,
                    sources=context['content'],
                    metadata={
                        'question_type': question.type,
                        'generated_at': datetime.utcnow().isoformat(),
                        'context_query': context.get('query'),
                        'key_terms': context.get('terms'),
                        'confidence_metrics': metrics.__dict__ if metrics else None
                    }
                )
                logger.info(f"Successfully generated answer for question {question.id}")
                return answer

            except Exception as e:
                logger.error(f"Error creating Answer object: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Error in _generate_answer for question {question.id}: {str(e)}")
            raise

    async def _generate_answer_old(self, question: Question, context: Dict) -> Answer:
        """Generate answer for a question using context"""
        try:
            # Create prompt based on question type and complexity
            prompt = self._create_prompt(question, context)

            # Generate answer using LLM and track generation time
            start_time = time.time()
            response = await self.llm.agenerate(prompt)
            generation_time = time.time() - start_time

            # Add generation time to metadata
            response_metadata = {
                'generation_time': generation_time,
                'token_usage': response.usage.total_tokens if hasattr(response, 'usage') else None,
                'model_confidence': getattr(response, 'model_confidence', None)
            }

            # Calculate confidence score
            confidence_score, metrics = self.confidence_scorer.calculate_confidence(
                question=question.text,
                answer=response.content,
                context=context['content'],
                response_metadata=response_metadata,
                previous_answers=self.previous_answers
            )

            # Create answer object
            answer = Answer(
                text=response.content,
                confidence=confidence_score,
                sources=context['content'],
                metadata={
                    'question_type': question.type,
                    'generated_at': datetime.now().isoformat(),
                    'context_query': context['query'],
                    'key_terms': context['terms'],
                    'confidence_metrics': metrics.__dict__ if metrics else None,
                    'generation_metrics': response_metadata
                }
            )

            # Store answer for future consistency checking
            self.previous_answers[question.id] = response.content

            return answer

        except Exception as e:
            logger.error(f"Error generating answer for question {question.id}: {str(e)}")
            raise

    def _create_prompt(self, question: Question, context: Dict) -> str:
        """Create appropriate prompt based on question type"""
        # Base prompt template
        prompt = f"""Answer this DDQ question accurately and professionally.
Question: {question.text}
Category: {question.metadata.category}
Context: {self._format_context(context['content'])}

Guidelines:
- Answer directly and factually
- Include specific details from the context
- Format appropriately for a {question.type} question
- Maintain professional tone
- Be concise but complete

Answer:"""
        return prompt

    def _format_context(self, context: List[Dict]) -> str:
        """Format context for prompt"""
        return "\n\n".join([doc['content'] for doc in context])




