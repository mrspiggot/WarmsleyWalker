# File: ddqpro/app.py
import streamlit as st
from pathlib import Path
from typing import List, Optional, Dict
import logging
from ddqpro.models.state import DDQState
from ddqpro.rag.document_processor import CorpusProcessor
from ddqpro.graphs.workflow import build_workflow
from ddqpro.gui.components.sidebar import Sidebar
from ddqpro.gui.components.upload import FileUploader, UploadedFile
from ddqpro.gui.components.progress import ProgressTracker, ProcessingStage
import time
from ddqpro.models.llm_manager import LLMManager
from ddqpro.rag.retriever import RAGRetriever
from ddqpro.agents.enhanced_processor import EnhancedProcessor

logger = logging.getLogger(__name__)

class DDQApp:
    def __init__(self):
        self.setup_page()
        # Initialize the corpus processor with the proper corpus directory.
        self.corpus_processor = CorpusProcessor(corpus_dir="ddqpro/gui/data/corpus")
        self.workflow = build_workflow()
        self.sidebar = Sidebar()
        self.file_uploader = FileUploader()
        self.progress_tracker = ProgressTracker()

    def setup_page(self):
        st.set_page_config(page_title="DDQ Processor", page_icon="📋", layout="wide")
        st.title("DDQ Document Processor")

    def run(self):
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'results' not in st.session_state:
            st.session_state.results = None

        col1, col2 = st.columns([2, 1])
        with col1:
            self.render_main_content()
        with col2:
            self.render_sidebar()

    def render_main_content(self):
        corpus_files, ddq_file = self.file_uploader.render_upload_sections()
        button_disabled = not (corpus_files or ddq_file)
        if st.button("Process DDQ", key='process_button', disabled=button_disabled, help="Upload files before processing"):
            self.process_files(corpus_files, ddq_file)
        if st.session_state.processing_complete:
            self.display_results()

    def render_sidebar(self):
        provider, model_name, config = self.sidebar.render()
        st.session_state.provider = provider
        st.session_state.model_name = model_name
        st.session_state.config = config

        if provider and model_name:
            try:
                llm_manager = LLMManager()
                llm_manager.initialize(provider, model_name)
                logger.info(f"Initialized LLM with {provider} {model_name}")
            except Exception as e:
                logger.error(f"Error initializing LLM: {str(e)}")
                st.error(f"Error initializing LLM: {str(e)}")
        return provider, model_name

    def process_files(self, corpus_files: List[UploadedFile], ddq_file: Optional[UploadedFile]):
        print("\n=== Processing Files ===")
        print(f"Corpus files: {[f.name for f in corpus_files] if corpus_files else 'None'}")
        print(f"DDQ file: {ddq_file.name if ddq_file else 'None'}")
        containers = self.progress_tracker.create_progress_container()
        st.session_state.start_time = time.time()

        try:
            if corpus_files:
                self.progress_tracker.update_progress(ProcessingStage.CORPUS, "Loading documents")
                for idx, file in enumerate(corpus_files):
                    try:
                        progress = (idx + 1) / len(corpus_files)
                        self.progress_tracker.update_progress(ProcessingStage.CORPUS, f"Processing {file.name}", progress)
                        self.process_corpus_files([file])
                    except Exception as e:
                        self.progress_tracker.error(containers, f"Error processing {file.name}: {str(e)}")
                        return

            if ddq_file:
                self.progress_tracker.update_progress(ProcessingStage.ANALYSIS, "Starting document analysis")
                try:
                    # Build a retriever using the shared vector store from corpus_processor.
                    shared_retriever = RAGRetriever(vector_store=self.corpus_processor.vector_store)
                    enhanced_processor = EnhancedProcessor(retriever=shared_retriever)
                    state = self.process_ddq_with_progress(ddq_file, containers, enhanced_processor)
                    st.session_state.results = state.get('json_output')
                    self.progress_tracker.success(containers)
                    st.session_state.processing_complete = True
                except Exception as e:
                    self.progress_tracker.error(containers, f"Error processing DDQ: {str(e)}")
                    return
            self.progress_tracker.render_progress(containers)
        except Exception as e:
            self.progress_tracker.error(containers, str(e))

    def process_corpus_files(self, files: List[UploadedFile]):
        logger.info("\n=== Processing Corpus Files ===")
        try:
            logger.info("Initializing corpus processor...")
            logger.info(f"Corpus directory: {self.corpus_processor.corpus_dir}")
            logger.info(f"Corpus dir exists: {self.corpus_processor.corpus_dir.exists()}")
            for file in files:
                try:
                    logger.info(f"\nProcessing file: {file.name}")
                    logger.debug(f"File type: {type(file)}")
                    file_path = Path("ddqpro/gui/data/corpus") / file.name
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Saving to: {file_path.absolute()}")
                    with open(file_path, "wb") as f:
                        f.write(file.content)
                    logger.info("Ingesting documents...")
                    self.corpus_processor.ingest_documents()
                    logger.info("Document ingestion complete")
                except Exception as e:
                    logger.error(f"Error processing {file.name}: {str(e)}", exc_info=True)
                    st.error(f"Error processing {file.name}: {str(e)}")
        except Exception as e:
            logger.error(f"Corpus processing error: {str(e)}", exc_info=True)

    def process_ddq_with_progress(self, ddq_file: UploadedFile, containers: Dict, enhanced_processor: EnhancedProcessor) -> DDQState:
        try:
            state = DDQState(
                input_path=str(ddq_file.path),
                file_type=ddq_file.file_type,
                current_extractor="default",
                extraction_results=None,
                reflections=[],
                json_output=None,
                response_states={},
                completed_responses={},
                cost_tracking={}
            )
            self.progress_tracker.update_progress(ProcessingStage.EXTRACTION, "Extracting document content")
            final_state = self.workflow.invoke(state)
            self.progress_tracker.update_progress(ProcessingStage.COMPLETION, "Finalizing results", 1.0)
            return final_state
        except Exception as e:
            logger.error(f"Error in DDQ processing: {str(e)}")
            raise

    def process_ddq(self, ddq_file: UploadedFile):
        try:
            file_path = Path("ddqpro/gui/data/input") / ddq_file.name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(ddq_file.content)
            state = DDQState(
                input_path=str(file_path),
                file_type=ddq_file.file_type,
                current_extractor="default",
                extraction_results=None,
                reflections=[],
                json_output=None,
                response_states={},
                completed_responses={},
                cost_tracking={}
            )
            final_state = self.workflow.invoke(state)
            st.session_state.results = final_state.get('json_output')
        except Exception as e:
            st.error(f"Error processing DDQ: {str(e)}")
            logger.error(f"DDQ processing error: {str(e)}", exc_info=True)

    def display_results(self):
        if st.session_state.results:
            st.header("3. Results")
            st.json(st.session_state.results)

def main():
    app = DDQApp()
    app.run()

if __name__ == "__main__":
    main()
