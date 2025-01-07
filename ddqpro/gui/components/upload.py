# ddqpro/gui/components/upload.py

import streamlit as st
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class UploadedFile:
    name: str
    content: bytes
    file_type: str
    path: Path


class FileUploader:
    def __init__(self):
        self.corpus_dir = Path("data/corpus")
        self.input_dir = Path("data/input")
        self.supported_extensions = {
            'corpus': ['.pdf', '.docx'],
            'ddq': ['.pdf']
        }

    def render_upload_sections(self) -> Tuple[List[UploadedFile], Optional[UploadedFile]]:
        """Render both upload sections and return processed files"""
        corpus_files = self.render_corpus_uploader()
        ddq_file = self.render_ddq_uploader()

        return corpus_files, ddq_file

    def render_corpus_uploader(self) -> List[UploadedFile]:
        """Render corpus documents upload section"""
        st.header("1. Build Document Corpus")

        # Information expander
        with st.expander("ℹ️ About the Document Corpus", expanded=False):
            st.markdown("""
            The document corpus is used to provide context for answering DDQ questions. You should upload:
            - Previous DDQ responses
            - Investment policies
            - Operational documents
            - Any other relevant reference materials
            """)

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload reference documents",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            key='corpus_files',
            help="Upload PDF or DOCX files containing reference materials"
        )

        if uploaded_files:
            processed_files = self._process_corpus_files(uploaded_files)
            self._show_corpus_summary(processed_files)
            return processed_files
        return []

    def render_ddq_uploader(self) -> Optional[UploadedFile]:
        """Render DDQ document upload section"""
        st.header("2. Process DDQ")

        with st.expander("ℹ️ About DDQ Processing", expanded=False):
            st.markdown("""
            Upload a single DDQ document to be processed. The system will:
            1. Extract questions and structure
            2. Generate responses using the document corpus
            3. Produce a standardized JSON output
            """)

        uploaded_file = st.file_uploader(
            "Upload DDQ document",
            type=['pdf'],
            key='ddq_file',
            help="Upload a PDF file containing the DDQ to be processed"
        )

        if uploaded_file:
            processed_file = self._process_ddq_file(uploaded_file)
            self._show_ddq_summary(processed_file)
            return processed_file
        return None

    def _process_corpus_files(self, files: List) -> List[UploadedFile]:
        """Process uploaded corpus files"""
        processed_files = []

        for file in files:
            try:
                logger.debug(f"Processing file: {file.name}, type: {type(file)}")

                # Read the file content using Streamlit's UploadedFile interface
                file_content = file.read()

                file_path = self.corpus_dir / file.name
                self.corpus_dir.mkdir(parents=True, exist_ok=True)

                logger.debug(f"Saving file to: {file_path}")

                # Create UploadedFile object with the correct content
                processed_file = UploadedFile(
                    name=file.name,
                    content=file_content,
                    file_type=Path(file.name).suffix.lower(),
                    path=file_path
                )
                processed_files.append(processed_file)

                # Save file to disk
                logger.debug(f"Writing {len(file_content)} bytes to disk")
                with open(file_path, "wb") as f:
                    f.write(file_content)

                logger.info(f"Successfully processed corpus file: {file.name}")

                # Seek back to start of file for any subsequent reads
                file.seek(0)

            except Exception as e:
                logger.error(f"Error processing corpus file {file.name}: {str(e)}", exc_info=True)
                st.error(f"Error processing {file.name}: {str(e)}")

        return processed_files

    def _process_ddq_file(self, file) -> Optional[UploadedFile]:
        """Process uploaded DDQ file"""
        try:
            file_path = self.input_dir / file.name
            self.input_dir.mkdir(parents=True, exist_ok=True)

            # Create UploadedFile object
            processed_file = UploadedFile(
                name=file.name,
                content=file.read(),
                file_type=Path(file.name).suffix.lower(),
                path=file_path
            )

            # Save file to disk
            with open(file_path, "wb") as f:
                f.write(processed_file.content)

            logger.info(f"Successfully processed DDQ file: {file.name}")
            return processed_file

        except Exception as e:
            logger.error(f"Error processing DDQ file {file.name}: {str(e)}")
            st.error(f"Error processing {file.name}: {str(e)}")
            return None

    def _show_corpus_summary(self, files: List[UploadedFile]):
        """Display summary of processed corpus files"""
        with st.expander("📁 Corpus Files Summary", expanded=True):
            st.markdown("### Processed Files")

            # Create a summary table
            file_data = []
            for file in files:
                file_size = len(file.content) / (1024 * 1024)  # Convert to MB
                file_data.append({
                    "File Name": file.name,
                    "Type": file.file_type,
                    "Size (MB)": f"{file_size:.2f}"
                })

            if file_data:
                st.table(file_data)
                st.markdown(f"**Total Files:** {len(files)}")
            else:
                st.info("No files processed yet")

    def _show_ddq_summary(self, file: UploadedFile):
        """Display summary of processed DDQ file"""
        with st.expander("📄 DDQ File Summary", expanded=True):
            if file:
                file_size = len(file.content) / (1024 * 1024)  # Convert to MB
                st.markdown(f"""
                ### DDQ Document Details
                - **File Name:** {file.name}
                - **File Type:** {file.file_type}
                - **Size:** {file_size:.2f} MB
                - **Status:** Ready for processing
                """)
            else:
                st.info("No DDQ file processed yet")