from typing import Dict
from pathlib import Path
from ddqpro.models.state import DDQState, ExtractionResult
from ddqpro.tools.document_tools import PDFExtractor, DocxExtractor
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import logging
from ddqpro.models.llm_manager import LLMManager

logger = logging.getLogger(__name__)

# Note the double curly braces for JSON structure
ANALYSIS_TEMPLATE = """You are an expert at analyzing Due Diligence Questionnaires (DDQs). 
Your task is to identify questions and transform them into a structured JSON format.

Instructions:
1. Identify all sections in the document
2. Extract all questions within each section
3. Count total sections and questions
4. Assign appropriate metadata to each question

Output format must be EXACTLY as shown below (replace placeholders with actual values):
{{
    "section_count": 3,
    "total_question_count": 9,
    "sections": [
        {{
            "title": "SECTION NAME",
            "level": 1,
            "questions": [
                {{
                    "id": "1.1",
                    "text": "What is your firm's legal name?",
                    "type": "text",
                    "required": true,
                    "metadata": {{
                        "category": "general",
                        "subcategory": "identification",
                        "context": "basic information"
                    }}
                }}
            ]
        }}
    ],
    "completeness_score": 0.95,
    "potential_issues": ["Some questions lack clear categories"]
}}

Content to analyze: {content}

Remember: Your response must be VALID JSON and match the exact structure shown above.
Do not include any additional text before or after the JSON structure.
"""

class DocumentAnalyzer:
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.docx_extractor = DocxExtractor()
        self.llm_manager = LLMManager()

        # Setup prompt for document analysis
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", ANALYSIS_TEMPLATE),
            ("user", "{content}")
        ])

        self.parser = JsonOutputParser()

    def analyze(self, state: DDQState) -> DDQState:
        """Analyze the document structure and content"""
        print("\n=== Starting analyze method ===")
        analysis_result = None

        try:
            print("1. Starting document extraction")
            if state['file_type'] == '.pdf':
                print("2. Extracting PDF")
                extraction = self.pdf_extractor.extract(state['input_path'])
            elif state['file_type'] == '.docx':
                extraction = self.docx_extractor.extract(state['input_path'])
            else:
                raise ValueError(f"Unsupported file type: {state['file_type']}")

            print("3. Checking extraction success")
            if not extraction.get('success'):
                raise Exception(f"Failed to extract document: {extraction.get('error')}")

            print(f"4. Extraction successful, content length: {len(extraction['raw_content'])} characters")
            logger.info(f"Extracted content length: {len(extraction['raw_content'])} characters")
            logger.debug(f"Content preview: {extraction['raw_content'][:500]}...")

            print("5. Creating LLM chain")
            chain = self.analysis_prompt | self.llm_manager.llm | self.parser

            print("6. About to invoke chain")
            try:
                print("7. Invoking chain with content")
                analysis_result = chain.invoke({"content": extraction['raw_content']})
                print("8. Chain response received")
                print(f"9. Response type: {type(analysis_result)}")
                print(f"10. Response preview: {str(analysis_result)[:200]}")

                # Extract all questions into a flat structure for easier access
                print("11. Processing response into questions")
                questions_flat = {}
                print("12. Analysis Result sections:", len(analysis_result['sections']))

                for section in analysis_result['sections']:
                    print(f"13. Processing section: {section['title']}")
                    for question in section['questions']:
                        questions_flat[question['id']] = {
                            **question,
                            'section': section['title']
                        }
                print(f"14. Total Questions processed: {len(questions_flat)}")

                # Create ExtractionResult object
                print("15. Creating ExtractionResult")
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

                print("16. Creating final JSON output")
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

                print("17. Updating state with results")
                return {
                    **state,
                    'extraction_results': extraction_result,
                    'json_output': json_output
                }

            except Exception as e:
                print("ERROR in chain processing:", str(e))
                print("analysis_result at error:", analysis_result)
                error_msg = f"Error in LLM analysis: {str(e)}"
                if analysis_result is not None:
                    error_msg += f"\nRaw LLM response: {analysis_result}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        except Exception as e:
            print("FINAL ERROR in analyze method:", str(e))
            print("Final analysis_result state:", analysis_result)
            logger.error(f"Error analyzing document: {str(e)}")
            raise