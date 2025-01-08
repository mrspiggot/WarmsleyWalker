# app.py

import streamlit as st
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import logging
from ddqpro.models.state import DDQState
from ddqpro.rag.document_processor import CorpusProcessor
from ddqpro.graphs.workflow import build_workflow
from ddqpro.gui.components.sidebar import Sidebar
from ddqpro.gui.components.upload import FileUploader, UploadedFile
from ddqpro.gui.components.progress import ProgressTracker, ProcessingStage
import time
from ddqpro.gui.components.upload import UploadedFile  # Import our custom class
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DDQApp:
    def __init__(self):
        self.setup_page()
        self.corpus_processor = CorpusProcessor(
            corpus_dir="ddqpro/gui/data/corpus",

        )
        self.workflow = build_workflow()
        self.sidebar = Sidebar()
        self.file_uploader = FileUploader()
        self.progress_tracker = ProgressTracker()

    def setup_page(self):
        """Configure the Streamlit page"""
        st.set_page_config(
            page_title="DDQ Processor",
            page_icon="📋",
            layout="wide"
        )
        st.title("DDQ Document Processor")

    def run(self):
        """Main application loop"""
        # Initialize session state if needed
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'results' not in st.session_state:
            st.session_state.results = None

        # Create main page layout
        col1, col2 = st.columns([2, 1])

        with col1:
            self.render_main_content()

        with col2:
            self.render_sidebar()

    def render_main_content(self):
        """Render the main content area"""
        # Get processed files from uploader component
        corpus_files, ddq_file = self.file_uploader.render_upload_sections()

        # Process button - only enable if files are uploaded
        button_disabled = not (corpus_files or ddq_file)
        if st.button("Process DDQ",
                     key='process_button',
                     disabled=button_disabled,
                     help="Upload files before processing"):
            self.process_files(corpus_files, ddq_file)

        # Results section
        if st.session_state.processing_complete:
            self.display_results()

    def render_sidebar(self):
        """Render the sidebar with model selection"""
        provider, model_name, config = self.sidebar.render()

        # Store in session state
        st.session_state.provider = provider
        st.session_state.model_name = model_name
        st.session_state.config = config

        # Initialize the LLM
        if provider and model_name:
            try:
                from ddqpro.models.llm_manager import LLMManager
                llm_manager = LLMManager()
                llm_manager.initialize(provider, model_name)
                logger.info(f"Initialized LLM with {provider} {model_name}")
            except Exception as e:
                logger.error(f"Error initializing LLM: {str(e)}")
                st.error(f"Error initializing LLM: {str(e)}")

        return provider, model_name

    def get_available_models(self, provider: str) -> List[str]:
        """Get available models for the selected provider"""
        models = {
            "OpenAI": [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo-preview",
                "gpt-3.5-turbo"
            ],
            "Anthropic": [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ],
            "Google": [
                "gemini-1.5-pro",
                "gemini-1.0-pro"
            ]
        }
        return models.get(provider, [])

    def process_files(self, corpus_files: List[UploadedFile], ddq_file: Optional[UploadedFile]):
        """Process uploaded files with progress tracking"""
        # Create progress containers
        containers = self.progress_tracker.create_progress_container()
        st.session_state.start_time = time.time()

        try:
            # Process corpus files if provided
            if corpus_files:
                self.progress_tracker.update_progress(
                    ProcessingStage.CORPUS,
                    "Loading documents"
                )

                for idx, file in enumerate(corpus_files):
                    try:
                        # Update progress for each file
                        progress = (idx + 1) / len(corpus_files)
                        self.progress_tracker.update_progress(
                            ProcessingStage.CORPUS,
                            f"Processing {file.name}",
                            progress
                        )
                        self.process_corpus_files([file])

                    except Exception as e:
                        self.progress_tracker.error(containers, f"Error processing {file.name}: {str(e)}")
                        return

            # Process DDQ if provided
            if ddq_file:
                # Analysis stage
                self.progress_tracker.update_progress(
                    ProcessingStage.ANALYSIS,
                    "Starting document analysis"
                )

                try:
                    # Process DDQ with progress updates
                    state = self.process_ddq_with_progress(ddq_file, containers)
                    st.session_state.results = state.get('json_output')

                    # Complete processing
                    self.progress_tracker.success(containers)
                    st.session_state.processing_complete = True

                except Exception as e:
                    self.progress_tracker.error(containers, f"Error processing DDQ: {str(e)}")
                    return

            self.progress_tracker.render_progress(containers)

        except Exception as e:
            self.progress_tracker.error(containers, str(e))

    def process_ddq_with_progress(self, ddq_file: UploadedFile, containers: Dict) -> DDQState:
        """Process DDQ with progress updates"""
        try:
            # Initial state
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

            # Update progress through workflow stages
            self.progress_tracker.update_progress(
                ProcessingStage.EXTRACTION,
                "Extracting document content"
            )

            # Process using workflow with progress updates
            final_state = self.workflow.invoke(state)

            # Final progress update
            self.progress_tracker.update_progress(
                ProcessingStage.COMPLETION,
                "Finalizing results",
                1.0
            )

            return final_state

        except Exception as e:
            logger.error(f"Error in DDQ processing: {str(e)}")
            raise

    def process_corpus_files(self, files: List[UploadedFile]):  # Add type hint
        """Process corpus files"""
        for file in files:
            try:
                logger.debug(f"Processing file: {file.name} of type {type(file)}")

                # Save uploaded file temporarily
                file_path = Path("ddqpro/gui/data/corpus") / file.name
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Use the content directly from our custom UploadedFile
                with open(file_path, "wb") as f:
                    f.write(file.content)  # Use .content instead of .read() or .getvalue()

                # Process the file
                self.corpus_processor.ingest_documents()

            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                logger.error(f"Corpus processing error: {str(e)}", exc_info=True)

    def process_ddq(self, ddq_file: UploadedFile):  # Add type hint
        """Process DDQ document"""
        try:
            # Save uploaded file temporarily
            file_path = Path("ddqpro/gui/data/input") / ddq_file.name
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Use the content directly from our custom UploadedFile
            with open(file_path, "wb") as f:
                f.write(ddq_file.content)  # Use .content instead of .getvalue()

            # Create initial state
            state = DDQState(
                input_path=str(file_path),
                file_type=ddq_file.file_type,  # Use the file_type from our UploadedFile
                current_extractor="default",
                extraction_results=None,
                reflections=[],
                json_output=None,
                response_states={},
                completed_responses={},
                cost_tracking={}
            )

            # Process using workflow
            final_state = self.workflow.invoke(state)
            st.session_state.results = final_state.get('json_output')

        except Exception as e:
            st.error(f"Error processing DDQ: {str(e)}")
            logger.error(f"DDQ processing error: {str(e)}", exc_info=True)

    def display_results(self):
        """Display processing results"""
        if st.session_state.results:
            st.header("3. Results")
            st.json(st.session_state.results)


def main():
    app = DDQApp()
    app.run()


if __name__ == "__main__":
    main()