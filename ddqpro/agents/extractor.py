from typing import Dict, List
import asyncio
import logging
from ddqpro.models.state import DDQState, Answer
from ddqpro.agents.processor import SectionProcessor

logger = logging.getLogger(__name__)


class DocumentExtractor:
    def __init__(self):
        self.processor = SectionProcessor()

    def extract(self, state: DDQState) -> DDQState:
        """Extract content and generate answers"""
        logger.info(f"Extracting content from: {state['input_path']}")

        if not state.get('extraction_results'):
            logger.warning("No extraction results found in state")
            return state

        try:
            # Group questions by section
            sections = self._group_by_section(state['extraction_results'].content['questions'])

            # Process all sections concurrently
            answers = asyncio.run(self._process_sections(sections))

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

    def _group_by_section(self, questions: Dict) -> Dict[str, List]:
        """Group questions by section"""
        sections = {}
        for qid, question in questions.items():
            section = question['section']
            if section not in sections:
                sections[section] = []
            sections[section].append(question)
        return sections

    async def _process_sections(self, sections: Dict[str, List]) -> Dict[str, Answer]:
        """Process all sections concurrently"""
        tasks = []
        for section_name, questions in sections.items():
            batch_size = self._get_batch_size(section_name)
            task = self.processor.process_section(section_name, questions, batch_size)
            tasks.append(task)

        # Process all sections concurrently
        section_results = await asyncio.gather(*tasks)

        # Combine all results
        all_answers = {}
        for result in section_results:
            all_answers.update(result)
        return all_answers

    def _get_batch_size(self, section_name: str) -> int:
        """Determine optimal batch size for section"""
        batch_sizes = {
            "MANAGER INFORMATION": 20,  # Simple questions
            "FUND INFORMATION": 15,  # Medium complexity
            "STRATEGY": 5,  # Complex questions
            "RISK": 8,  # Medium-complex
            "DEFAULT": 10  # Default batch size
        }
        return batch_sizes.get(section_name.upper(), batch_sizes["DEFAULT"])