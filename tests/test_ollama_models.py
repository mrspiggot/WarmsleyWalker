
# File: tests/test_ollama_models.py

import os
from pathlib import Path
import json
import logging
import asyncio
from typing import List, Dict

from ddqpro.models.llm_manager import LLMManager
from ddqpro.models.state import DDQState
from ddqpro.rag.document_processor import CorpusProcessor
from ddqpro.graphs.workflow import build_workflow
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DDQModelTester:
    def __init__(self):
        self.models = [


            "mixtral:8x22b",
            "llama3.1:70b",
            "TiagoG/json-response",

        ]

        # Get project root directory (where the test file is located)
        self.project_root = Path(__file__).parent.parent

        # Setup directories with absolute paths
        self.ww_dir = self.project_root / "data" / "ww"
        self.sample_dir = self.project_root / "data" / "sample"
        self.output_dir = self.project_root / "data" / "output"

        logger.info(f"Project root: {self.project_root}")

        # Create directories if they don't exist
        self.ww_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize corpus processor with absolute paths
        self.corpus_processor = CorpusProcessor(
            corpus_dir=str(self.ww_dir.absolute()),
            db_dir=str((self.project_root / "data" / "vectordb").absolute())
        )

    # File: tests/test_ollama_models.py

    def check_prerequisites(self) -> bool:
        """Check if all required files and directories are present"""
        # Debug directory paths
        logger.info(f"Checking directories:")
        logger.info(f"WW directory: {self.ww_dir.absolute()}")
        logger.info(f"Sample directory: {self.sample_dir.absolute()}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")

        # Check RAG documents with detailed logging
        ww_files = list(self.ww_dir.glob('*.pdf'))
        logger.info(f"Found PDF files in {self.ww_dir}:")
        for file in ww_files:
            logger.info(f"  - {file.name}")

        if not ww_files:
            logger.error(f"No PDF files found in {self.ww_dir}. Please add RAG context PDFs.")
            return False

        # Check sample DDQ
        ddq_files = list(self.sample_dir.glob('*.pdf'))
        if not ddq_files:
            logger.error(f"No DDQ file found in {self.sample_dir}. Please add a sample DDQ.")
            return False

        logger.info(f"Found {len(ww_files)} RAG documents and {len(ddq_files)} DDQ files")
        return True

    def prepare_rag_context(self):
        """Process RAG documents"""
        logger.info("Processing RAG context documents")
        try:
            # Check if ww directory has files with detailed logging
            ww_files = list(self.ww_dir.glob('*.pdf'))
            logger.info(f"Full path being checked: {self.ww_dir.absolute()}")
            logger.info(f"Directory contents:")
            if self.ww_dir.exists():
                for item in self.ww_dir.iterdir():
                    logger.info(f"  - {item.name} ({'directory' if item.is_dir() else 'file'})")
            else:
                logger.error(f"Directory does not exist: {self.ww_dir.absolute()}")

            if not ww_files:
                logger.error("No PDF files found in ww directory")
                raise ValueError("RAG context documents required")

            logger.info(f"Found {len(ww_files)} PDF files:")
            for file in ww_files:
                logger.info(f"  - {file.name}")

            # Reset vector DB for clean test
            self.corpus_processor.ingest_documents(reset_db=True)
            logger.info("RAG context processing complete")
        except Exception as e:
            logger.error(f"Error processing RAG context: {str(e)}")
            raise

    def save_json_output(self, data: Dict, filename: str):
        """Save JSON output with proper formatting"""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved output to {output_path}")

    def process_ddq(self, model_name: str) -> None:
        """Process DDQ with specified model"""
        logger.info(f"\nProcessing DDQ with model: {model_name}")

        try:
            # Initialize LLM with no temperature (will be set by provider)
            llm_manager = LLMManager()
            llm_manager.initialize("ollama", model_name)

            # Build workflow without temperature
            workflow = build_workflow("ollama", model_name)

            # Find first DDQ file in sample directory
            ddq_files = list(self.sample_dir.glob('*.pdf'))
            if not ddq_files:
                raise ValueError("No DDQ files found in sample directory")
            ddq_file = ddq_files[0]

            # Create initial state
            state = DDQState(
                input_path=str(ddq_file),
                file_type='.pdf',
                current_extractor="default",
                extraction_results=None,
                reflections=[],
                json_output=None,
                response_states={},
                completed_responses={},
                cost_tracking={}
            )

            # Process document
            logger.info(f"Processing {ddq_file.name}")
            final_state = workflow.invoke(state)

            # Save questions JSON
            if final_state.get('extraction_results'):
                questions_output = {
                    'metadata': {
                        'model': model_name,
                        'file': ddq_file.name
                    },
                    'sections': final_state['extraction_results'].content['sections'],
                    'questions': final_state['extraction_results'].content['questions']
                }
                self.save_json_output(
                    questions_output,
                    f"questions_{model_name.replace('/', '_')}.json"
                )
                logger.info(f"Saved questions output for {model_name}")

            # Save answers JSON if available
            if final_state.get('json_output'):
                self.save_json_output(
                    final_state['json_output'],
                    f"answers_{model_name.replace('/', '_')}.json"
                )
                logger.info(f"Saved answers output for {model_name}")

            logger.info(f"Processing complete for model: {model_name}")

        except Exception as e:
            logger.error(f"Error processing with model {model_name}: {str(e)}")
            raise

    def run_tests(self):
        """Run tests for all models"""
        logger.info("Starting DDQ model tests")

        try:
            # Prepare RAG context first
            self.prepare_rag_context()

            # Process with each model
            for model in self.models:
                try:
                    self.process_ddq(model)
                except Exception as e:
                    logger.error(f"Test failed for model {model}: {str(e)}")
                    continue

            logger.info("All tests complete")

        except Exception as e:
            logger.error(f"Test suite failed: {str(e)}")
            raise


def main():
    tester = DDQModelTester()
    tester.run_tests()


if __name__ == "__main__":
    main()
