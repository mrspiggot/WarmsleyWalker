from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ddqpro.models.state import DDQState, Answer, Question
from ddqpro.rag.retriever import RAGRetriever
from ddqpro.utils.cost_tracking import CostTracker
from ddqpro.models.llm_manager import LLMManager


logger = logging.getLogger(__name__)


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

    async def process_question(self,
                               section_name: str,
                               question: Question) -> Answer:
        """Process a single question"""
        try:
            # Get cached section context
            context = self.context_cache.get(section_name, [])

            # Get question-specific context if needed
            if self._needs_additional_context(question):
                specific_context = await self.retriever.aget_relevant_context(
                    question=question.text,
                    metadata_filter={'doc_type': 'DDQ'}
                )
                context.extend(specific_context)

            # Generate answer
            response = await self._generate_answer(question, context)

            return Answer(
                text=response.content,
                confidence=0.85,  # TODO: Implement confidence scoring
                sources=context,
                metadata={
                    "question_type": question.type,
                    "generated_at": datetime.now().isoformat()
                }
            )

        except Exception as e:
            logger.error(f"Error processing question {question.id}: {str(e)}")
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