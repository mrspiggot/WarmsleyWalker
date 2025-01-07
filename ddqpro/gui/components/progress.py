# ddqpro/gui/components/progress.py

import streamlit as st
from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Enum for different processing stages"""
    CORPUS = "corpus_processing"
    ANALYSIS = "ddq_analysis"
    EXTRACTION = "text_extraction"
    QUESTION_PROCESSING = "question_processing"
    RESPONSE_GENERATION = "response_generation"
    COMPLETION = "completion"


@dataclass
class StageConfig:
    """Configuration for a processing stage"""
    title: str
    description: str
    weight: float  # Percentage of total progress (0-1)
    substeps: List[str] = None


class ProgressTracker:
    def __init__(self):
        self.stages = {
            ProcessingStage.CORPUS: StageConfig(
                title="Processing Corpus Documents",
                description="Analyzing and embedding reference materials",
                weight=0.2,
                substeps=[
                    "Loading documents",
                    "Extracting text",
                    "Creating embeddings",
                    "Updating vector database"
                ]
            ),
            ProcessingStage.ANALYSIS: StageConfig(
                title="Analyzing DDQ Document",
                description="Extracting structure and questions",
                weight=0.2,
                substeps=[
                    "Document preprocessing",
                    "Structure analysis",
                    "Question identification",
                    "Metadata extraction"
                ]
            ),
            ProcessingStage.EXTRACTION: StageConfig(
                title="Extracting Content",
                description="Processing document content",
                weight=0.1,
                substeps=[
                    "Text extraction",
                    "Content validation",
                    "Format conversion"
                ]
            ),
            ProcessingStage.QUESTION_PROCESSING: StageConfig(
                title="Processing Questions",
                description="Analyzing and categorizing questions",
                weight=0.2,
                substeps=[
                    "Question categorization",
                    "Context retrieval",
                    "Response planning"
                ]
            ),
            ProcessingStage.RESPONSE_GENERATION: StageConfig(
                title="Generating Responses",
                description="Creating and validating responses",
                weight=0.25,
                substeps=[
                    "Context analysis",
                    "Response generation",
                    "Quality validation",
                    "Source attribution"
                ]
            ),
            ProcessingStage.COMPLETION: StageConfig(
                title="Completing Processing",
                description="Finalizing and formatting output",
                weight=0.05,
                substeps=[
                    "Response validation",
                    "Format conversion",
                    "Output generation"
                ]
            )
        }

        self.current_stage = None
        self.stage_progress = {}
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'processing_progress' not in st.session_state:
            st.session_state.processing_progress = 0.0
        if 'current_status' not in st.session_state:
            st.session_state.current_status = ""
        if 'stage_status' not in st.session_state:
            st.session_state.stage_status = {}
        if 'error_message' not in st.session_state:
            st.session_state.error_message = None

    def create_progress_container(self):
        """Create and return containers for progress display"""
        progress_container = st.empty()
        status_container = st.empty()
        detail_container = st.empty()
        error_container = st.empty()

        return {
            'progress': progress_container,
            'status': status_container,
            'detail': detail_container,
            'error': error_container
        }

    def update_progress(self,
                        stage: ProcessingStage,
                        substep: Optional[str] = None,
                        progress: float = None,
                        status_text: Optional[str] = None,
                        error: Optional[str] = None):
        """Update progress for a stage"""
        try:
            if error:
                logger.error(f"Error in {stage.value}: {error}")
                st.session_state.error_message = error
                return

            # Update stage progress
            if progress is not None:
                self.stage_progress[stage] = min(1.0, progress)
            elif substep:
                config = self.stages[stage]
                if config.substeps and substep in config.substeps:
                    step_index = config.substeps.index(substep)
                    self.stage_progress[stage] = (step_index + 1) / len(config.substeps)

            # Calculate overall progress
            total_progress = sum(
                self.stage_progress.get(s, 0) * self.stages[s].weight
                for s in ProcessingStage
            )

            st.session_state.processing_progress = total_progress
            st.session_state.current_status = status_text or self.stages[stage].description
            st.session_state.stage_status[stage] = substep or self.stages[stage].title

        except Exception as e:
            logger.error(f"Error updating progress: {str(e)}")
            st.session_state.error_message = f"Error updating progress: {str(e)}"

    def render_progress(self, containers: Dict):
        """Render progress display"""
        try:
            # Main progress bar
            containers['progress'].progress(st.session_state.processing_progress)

            # Status message
            if st.session_state.current_status:
                containers['status'].info(st.session_state.current_status)

            # Detailed progress
            with containers['detail'].expander("View Detailed Progress", expanded=True):
                for stage in ProcessingStage:
                    if stage in st.session_state.stage_status:
                        status = st.session_state.stage_status[stage]
                        progress = self.stage_progress.get(stage, 0)
                        st.markdown(f"**{self.stages[stage].title}** - {status}")
                        if progress > 0:
                            st.progress(progress)

            # Error message
            if st.session_state.error_message:
                containers['error'].error(st.session_state.error_message)

        except Exception as e:
            logger.error(f"Error rendering progress: {str(e)}")
            containers['error'].error(f"Error rendering progress: {str(e)}")

    def success(self, containers: Dict):
        """Display success message and final progress"""
        try:
            st.session_state.processing_progress = 1.0
            containers['progress'].progress(1.0)
            containers['status'].success("Processing completed successfully!")

            # Show completion time if available
            if hasattr(st.session_state, 'start_time'):
                duration = time.time() - st.session_state.start_time
                containers['detail'].info(f"Total processing time: {duration:.2f} seconds")

        except Exception as e:
            logger.error(f"Error displaying success: {str(e)}")
            containers['error'].error(f"Error displaying success: {str(e)}")

    def error(self, containers: Dict, error_message: str):
        """Display error message"""
        logger.error(f"Processing error: {error_message}")
        containers['error'].error(f"Error: {error_message}")