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
from ddqpro.utils.cost_tracking import CostTracker

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
    parser = argparse.ArgumentParser(
        description="""
DDQPro - AI-powered DDQ Processing Tool

This tool processes Due Diligence Questionnaires (DDQs) and generates standardized JSON responses.
It can also manage a corpus of documents for context-aware answer generation.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/Output options
    io_group = parser.add_argument_group('Input/Output Options')
    io_group.add_argument(
        '--input-dir',
        type=str,
        default='data/sample',
        help='Directory containing DDQ documents to process (default: data/sample)'
    )
    io_group.add_argument(
        '--output-dir',
        type=str,
        default='data/output',
        help='Directory for JSON output files (default: data/output)'
    )

    # Corpus management
    corpus_group = parser.add_argument_group('Corpus Management')
    corpus_group.add_argument(
        '--process-corpus',
        action='store_true',
        help='Process documents in data/corpus for RAG context'
    )
    corpus_group.add_argument(
        '--reset-db',
        action='store_true',
        help='Reset vector database before processing corpus'
    )
    corpus_group.add_argument(
        '--show-db-info',
        action='store_true',
        help='Show information about the current vector database'
    )

    # Processing options
    processing_group = parser.add_argument_group('Processing Options')
    processing_group.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing of files even if output exists'
    )
    processing_group.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of questions to process in parallel (default: 10)'
    )

    # Debug options
    debug_group = parser.add_argument_group('Debug Options')
    debug_group.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    debug_group.add_argument(
        '--show-costs',
        action='store_true',
        help='Show estimated API costs after processing'
    )

    return parser

def setup_argument_parser_old() -> argparse.ArgumentParser:
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
    parser.add_argument(
        '--process-corpus',
        action='store_true',
        help='Process documents in data/corpus for RAG'
    )
    parser.add_argument(
        '--reset-db',
        action='store_true',
        help='Reset vector database before processing corpus'
    )
    parser.add_argument(
        '--show-db-info',
        action='store_true',
        help='Show information about the vector database'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    return parser


def process_corpus(reset_db: bool = False):
    """Process all documents in the corpus directory"""
    logger.info("Processing corpus documents")
    processor = CorpusProcessor(
        corpus_dir="data/corpus",
        db_dir="data/vectordb"
    )

    if reset_db:
        logger.warning("Resetting vector database")

    processor.ingest_documents(reset_db=reset_db)


def show_db_info():
    """Show information about the vector database"""
    processor = CorpusProcessor(
        corpus_dir="data/corpus",
        db_dir="data/vectordb"
    )
    info = processor.get_database_info()

    if info["exists"]:
        logger.info(f"Vector database contains {info['document_count']} documents")
        logger.info(f"Processed files: {len(info['processed_files'])}")
        for file in info['processed_files']:
            logger.info(f"  - {file}")
    else:
        logger.info("No vector database exists")

def save_json_output(data: Dict, output_path: Path) -> None:
    """Save JSON output with proper formatting"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved output to {output_path}")


def process_documents(input_dir: str, output_dir: str):
    """Process all documents in the input directory"""
    workflow = build_workflow()
    cost_tracker = CostTracker()

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
                json_output=None,
                response_states={},
                completed_responses={},
                cost_tracking={'tracker': cost_tracker}
            )

            # Process document
            try:
                logger.info("Invoking workflow")
                final_state = workflow.invoke(state)

                logger.info("Workflow completed. Checking results...")
                if final_state.get('json_output'):
                    output_path = Path(output_dir) / f"{file_path.stem}.json"
                    logger.info(f"Saving output to {output_path}")

                    # Log the content being saved
                    logger.debug(f"JSON output content: {json.dumps(final_state['json_output'], indent=2)[:500]}...")

                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(final_state['json_output'], f, indent=2, ensure_ascii=False)

                    logger.info(f"Successfully saved output to {output_path}")
                else:
                    logger.error("No json_output in final state")
                    logger.debug(f"Final state keys: {list(final_state.keys())}")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)

    logger.info("\nProcessing Cost Report:")
    logger.info(cost_tracker.get_report())

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