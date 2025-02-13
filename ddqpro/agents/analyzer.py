# ddqpro/agents/analyzer.py

import logging
from pathlib import Path
from ddqpro.models.state import DDQState, ExtractionResult as StateExtractionResult  # Assuming we add extraction metadata here if needed
from ddqpro.extractors.pipeline import ExtractionPipeline
from ddqpro.extractors.pdf_extractor import PDFDirectExtractor
from ddqpro.extractors.ocr_extractor import TextractExtractor
from ddqpro.tools.document_tools import DocxExtractor  # Keeping docx extraction as before, if needed
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)

# Template for analysis
ANALYSIS_TEMPLATE = """You are an expert at analyzing Due Diligence Questionnaires (DDQs). 
Your task is to identify questions and transform them into a structured JSON format.

Instructions:
1. Identify all sections in the document.
2. Extract all questions within each section.
3. Count total sections and questions.
4. Assign appropriate metadata to each question.

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
        # Initialize the extraction pipeline with our two extractors:
        self.extraction_pipeline = ExtractionPipeline([
            PDFDirectExtractor(),
            TextractExtractor(
                region_name=None,  # will use environment default via load_dotenv()
                bucket_name=None  # will use environment default via load_dotenv()
            )
        ])
        self.docx_extractor = DocxExtractor()  # remains unchanged

        # Setup LLM for analysis as before
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.0
        )
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", ANALYSIS_TEMPLATE),
            ("user", "{content}")
        ])
        self.parser = JsonOutputParser()

    def analyze(self, state: DDQState) -> DDQState:
        """Analyze the document by extracting its text and processing with LLM"""
        logger.info("\n=== Starting document analysis ===")
        print(f"Input state: {state}")

        try:
            # Get file info
            file_type = state.get("file_type", "").lower()
            file_path = state.get("input_path")
            print(f"\nProcessing file: {file_path} (type: {file_type})")

            # Extract content
            extraction_result = None
            if file_type == ".pdf":
                extraction_result = self.extraction_pipeline.extract(file_path)
            elif file_type == ".docx":
                extraction_result = self.docx_extractor.extract(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            print(f"\nExtraction result: {extraction_result}")
            print(f"Extraction result type: {type(extraction_result)}")

            if not extraction_result.success:
                raise Exception(f"Extraction failed: {extraction_result.error}")

            # Process with LLM
            chain = self.analysis_prompt | self.llm | self.parser
            analysis_result = chain.invoke({"content": extraction_result.content})
            print(f"\nAnalysis result: {analysis_result}")

            # Update state
            state["extraction_results"] = extraction_result
            state["json_output"] = analysis_result

            print(f"\nFinal state: {state}")
            return state

        except Exception as e:
            logger.error(f"Error in document analysis: {str(e)}")
            print(f"Exception details: {str(e)}")
            raise

