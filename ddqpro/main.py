import argparse
import os
from pathlib import Path
import json
import logging
from typing import Dict

from dotenv import load_dotenv

from ddqpro.models.state import DDQState
from ddqpro.graphs.workflow import build_workflow
from ddqpro.rag.document_processor import CorpusProcessor

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)




def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Process DDQ documents into standardized JSON')
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/sample',
        help='Directory containing DDQ documents (default: data/sample)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/output',
        help='Directory for JSON output (default: data/output)'
    )
    # Add new argument for corpus processing
    parser.add_argument(
        '--process-corpus',
        action='store_true',
        help='Process documents in data/corpus for RAG'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    return parser
def process_corpus():
    """Process all documents in the corpus directory"""
    logger.info("Processing corpus documents")
    processor = CorpusProcessor(
        corpus_dir="data/corpus",
        db_dir="data/vectordb"
    )
    processor.ingest_documents()

def save_json_output(data: Dict, output_path: Path) -> None:
    """Save JSON output with proper formatting"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved output to {output_path}")


def process_documents(input_dir: str, output_dir: str):
    """Process all documents in the input directory"""
    workflow = build_workflow()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Processing documents from {input_dir}")

    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    # Process each document
    for file_path in Path(input_dir).glob('*'):
        if file_path.suffix.lower() in ['.docx', '.pdf']:
            logger.info(f"Processing {file_path}")

            # Initial state
            state = DDQState(
                input_path=str(file_path),
                file_type=file_path.suffix.lower(),
                current_extractor="default",
                extraction_results=None,
                reflections=[],
                json_output=None
            )

            # Process document
            try:
                final_state = workflow.invoke(state)

                # Save JSON output
                if final_state.get('json_output'):
                    output_path = Path(output_dir) / f"{file_path.stem}.json"
                    logger.info(f"Saving output to {output_path}")
                    with open(output_path, 'w') as f:
                        json.dump(final_state['json_output'], f, indent=2)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")

    logger.info("Document processing complete")
def process_documents_old(input_dir: str, output_dir: str):
    """Process all documents in the input directory"""
    workflow = build_workflow()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Processing documents from {input_dir}")

    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    # Process each document
    for file_path in input_path.glob('*'):
        if file_path.suffix.lower() in ['.docx', '.pdf']:
            logger.info(f"Starting processing of {file_path.name}")

            # Skip processing if output file already exists
            output_path = Path(output_dir) / f"{file_path.stem}.json"
            if output_path.exists():
                logger.info(f"Output file already exists for {file_path.name}, skipping...")
                continue

            # Initial state
            state = DDQState(
                input_path=str(file_path),
                file_type=file_path.suffix.lower(),
                current_extractor="default",
                extraction_results=None,
                reflections=[],
                json_output=None
            )

            # Process document
            try:
                logger.info(f"Analyzing document structure...")
                final_state = workflow.invoke(state)

                if final_state.get('extraction_results'):
                    logger.info(
                        f"Analysis complete for {file_path.name}:\n"
                        f"- Questions found: {final_state['extraction_results'].question_count}\n"
                        f"- Sections found: {final_state['extraction_results'].section_count}\n"
                        f"- Confidence score: {final_state['extraction_results'].confidence:.2f}"
                    )

                # Save JSON output
                if final_state.get('json_output'):
                    save_json_output(final_state['json_output'], output_path)
                else:
                    logger.warning(f"No JSON output generated for {file_path.name}")

            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {str(e)}", exc_info=True)
        else:
            logger.warning(f"Skipping unsupported file: {file_path.name}")

    logger.info("Document processing complete")


def main():
    parser = setup_argument_parser()
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.process_corpus:
        process_corpus()

    logger.info("Starting DDQ processing")
    process_documents(args.input_dir, args.output_dir)
    logger.info("DDQ processing complete")


if __name__ == "__main__":
    main()