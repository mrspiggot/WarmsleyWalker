
import asyncio
from typing import List, Dict
import logging
from ddqpro.rag.retriever import RAGRetriever
from ddqpro.models.state import Question, Answer
from ddqpro.agents.batch_processor import BatchProcessor
from ddqpro.utils.text_utils import extract_key_terms, create_search_query, group_by_similarity, COMMON_WORDS

logger = logging.getLogger(__name__)


class QuestionProcessor:
    def __init__(self, retriever: RAGRetriever, batch_processor: BatchProcessor):
        self.retriever = retriever
        self.batch_processor = batch_processor
        self.context_cache = {}

    async def process_questions(self,
                                questions: List[Question],
                                section: str) -> Dict[str, Answer]:
        """Process a list of questions efficiently"""
        logger.info(f"Processing {len(questions)} questions from section {section}")

        # Group similar questions
        question_groups = self._group_similar_questions(questions)

        # Process groups in parallel
        results = {}
        async with asyncio.TaskGroup() as group:
            tasks = []
            for group_id, group_questions in question_groups.items():
                task = group.create_task(
                    self._process_question_group(
                        questions=group_questions,
                        section=section,
                        group_id=group_id
                    )
                )
                tasks.append((group_id, task))

            # Collect results
            for group_id, task in tasks:
                group_results = await task
                results.update(group_results)

        return results

    def _group_similar_questions(self,
                                 questions: List[Question]) -> Dict[str, List[Question]]:
        """Group questions that might share context"""
        groups = {}
        for question in questions:
            # Group by category and type
            group_key = f"{question.metadata.category}_{question.type}"
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(question)
        return groups

    async def _process_question_group(self,
                                      questions: List[Question],
                                      section: str,
                                      group_id: str) -> Dict[str, Answer]:
        """Process a group of similar questions"""
        # Get shared context for the group
        shared_context = await self._get_group_context(questions, section, group_id)

        # Process questions using shared context
        async def process_single(question):
            return await self._generate_answer(
                question=question,
                shared_context=shared_context
            )

        results = await self.batch_processor.process_items(
            items=questions,
            section=section,
            processor_func=process_single
        )

        return {q.id: a for q, a in zip(questions, results)}

    async def _get_group_context(self,
                                 questions: List[Question],
                                 section: str,
                                 group_id: str) -> List[Dict]:
        """Get context shared by a group of questions"""
        cache_key = f"{section}_{group_id}"
        if cache_key not in self.context_cache:
            # Combine questions for context retrieval
            combined_query = self._create_combined_query(questions)
            context = await self.retriever.aget_relevant_context(
                question=combined_query,
                metadata_filter={'doc_type': 'DDQ'}
            )
            self.context_cache[cache_key] = context
        return self.context_cache[cache_key]

    def _create_combined_query(self, questions: List[Question]) -> str:
        """Create a combined query for a group of questions"""
        # Extract key terms and combine
        terms = set()
        for question in questions:
            terms.update(self._extract_key_terms(question))
        return " ".join(terms)

    def _extract_key_terms(self, question: Question) -> List[str]:
        """Extract key terms from a question"""
        # Simple implementation - could be enhanced with NLP
        text = question.text.lower()
        # Remove common words and punctuation
        # Return key terms
        return [word for word in text.split()
                if len(word) > 3 and word not in COMMON_WORDS]

    def _create_combined_query(self, questions: List[Question]) -> str:
        """Create a combined query for a group of questions"""
        # Extract key terms from all questions
        all_terms = set()
        for question in questions:
            all_terms.update(extract_key_terms(question.text))

        # Create optimized search query
        return create_search_query(all_terms)

    def _group_similar_questions(self,
                                 questions: List[Question]) -> Dict[str, List[Question]]:
        """Group questions that might share context"""
        # First group by category
        category_groups = {}
        for question in questions:
            category = question.metadata.category
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(question)

        # Then within each category, group by similarity
        final_groups = {}
        group_id = 0
        for category, cat_questions in category_groups.items():
            # Get question texts
            texts = [q.text for q in cat_questions]
            # Group similar questions
            similarity_groups = group_by_similarity(texts)

            # Create final groups
            for indices in similarity_groups.values():
                group_questions = [cat_questions[i] for i in indices]
                final_groups[f"group_{group_id}"] = group_questions
                group_id += 1

        return final_groups


