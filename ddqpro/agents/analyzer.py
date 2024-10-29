from typing import Dict
from pathlib import Path
from ddqpro.models.state import DDQState, ExtractionResult
from ddqpro.tools.document_tools import PDFExtractor, DocxExtractor
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import logging

logger = logging.getLogger(__name__)

ANALYSIS_TEMPLATE = """You are an expert at analyzing Due Diligence Questionnaires (DDQs).
Analyze the following document content and identify:
1. Key sections and their hierarchical structure
2. Questions within each section
3. Types of questions (multiple choice, text, tables, etc.)
4. Estimate of document completeness

Return your analysis in the following JSON format:
{{
    "section_count": int,
    "total_question_count": int,
    "sections": [
        {{
            "title": "section title",
            "level": int,
            "questions": [
                {{
                    "id": "unique question id (e.g., '1.1', '1.2', etc.)",
                    "text": "full question text",
                    "type": "text|multiple_choice|table",
                    "required": boolean,
                    "metadata": {{
                        "category": "risk|compliance|operations|etc",
                        "subcategory": "optional subcategory",
                        "context": "any relevant context or instructions"
                    }}
                }}
            ]
        }}
    ],
    "completeness_score": float,
    "potential_issues": ["list of potential issues"]
}}

answer as an employee of the firm. Do not speculate about missing information, do not start answers with equivocal 
phrases like: 'based on the provided context', rather answer the question as stated without embellishment sticking to 
the facts"""


class DocumentAnalyzer:
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.docx_extractor = DocxExtractor()
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

        # Setup prompt for document analysis
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", ANALYSIS_TEMPLATE),
            ("user", "{content}")
        ])

        self.parser = JsonOutputParser()

    def analyze(self, state: DDQState) -> DDQState:
        """Analyze the document structure and content"""
        logger.info(f"Analyzing document: {state['input_path']}")

        # Extract document content based on file type
        if state['file_type'] == '.pdf':
            extraction = self.pdf_extractor.extract(state['input_path'])
        elif state['file_type'] == '.docx':
            extraction = self.docx_extractor.extract(state['input_path'])
        else:
            raise ValueError(f"Unsupported file type: {state['file_type']}")

        if not extraction.get('success'):
            raise Exception(f"Failed to extract document: {extraction.get('error')}")

        logger.info(f"Extracted content length: {len(extraction['raw_content'])} characters")
        logger.debug(f"Content preview: {extraction['raw_content'][:500]}...")

        # Analyze the content using LLM
        chain = self.analysis_prompt | self.llm | self.parser

        try:
            analysis_result = chain.invoke({"content": extraction['raw_content']})

            # Extract all questions into a flat structure for easier access
            questions_flat = {}
            logger.info("Analysis Result:")
            logger.info(f"Sections: {len(analysis_result['sections'])}")
            for section in analysis_result['sections']:
                logger.info(f"Section '{section['title']}': {len(section['questions'])} questions")
                for question in section['questions']:
                    questions_flat[question['id']] = {
                        **question,
                        'section': section['title']
                    }
            logger.info(f"Total Questions: {len(questions_flat)}")

            # Create ExtractionResult object
            extraction_result = ExtractionResult(
                section_count=analysis_result['section_count'],
                question_count=analysis_result['total_question_count'],
                content={
                    'raw_content': extraction['raw_content'],
                    'sections': analysis_result['sections'],
                    'questions': questions_flat,
                    'potential_issues': analysis_result['potential_issues']
                },
                confidence=analysis_result['completeness_score']
            )

            logger.info(
                f"Analysis complete. Found {extraction_result.question_count} questions in {extraction_result.section_count} sections")

            # Prepare the JSON output
            json_output = {
                'metadata': {
                    'file_name': Path(state['input_path']).name,
                    'file_type': state['file_type'],
                    'extractor': state['current_extractor']
                },
                'analysis': {
                    'section_count': extraction_result.section_count,
                    'question_count': extraction_result.question_count,
                    'confidence_score': extraction_result.confidence,
                    'sections': analysis_result['sections'],
                    'questions': questions_flat
                },
                'potential_issues': analysis_result['potential_issues']
            }

            # Update state with both extraction results and JSON output
            return {
                **state,
                'extraction_results': extraction_result,
                'json_output': json_output
            }

        except Exception as e:
            logger.error(f"Error analyzing document: {str(e)}")
            raise