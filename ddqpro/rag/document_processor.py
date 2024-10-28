from typing import List, Dict, Optional
from pathlib import Path
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DocumentMetadata(BaseModel):
    """Metadata for processed documents"""
    source_file: str
    doc_type: str  # DDQ, LPA, etc.
    date_added: str
    chunk_index: int
    total_chunks: int


class CorpusProcessor:
    def __init__(self, corpus_dir: str, db_dir: str = "data/vectordb"):
        self.corpus_dir = Path(corpus_dir)
        self.db_dir = Path(db_dir)
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def ingest_documents(self, reset_db: bool = False) -> None:
        """Process all documents in the corpus directory

        Args:
            reset_db: If True, deletes existing vector database before processing
        """
        logger.info(f"Processing documents from {self.corpus_dir}")

        if reset_db:
            self._reset_vector_db()

        # Check if vector database already exists
        if self.db_dir.exists():
            logger.info(f"Vector database exists at {self.db_dir}")
            if not reset_db:
                logger.info("Adding new documents to existing database")

        documents = []
        processed_files = self._get_processed_files()

        for file_path in self.corpus_dir.glob('**/*'):
            try:
                # Skip already processed files unless resetting
                if not reset_db and str(file_path) in processed_files:
                    logger.info(f"Skipping already processed file: {file_path}")
                    continue

                if file_path.suffix.lower() == '.pdf':
                    logger.info(f"Processing PDF: {file_path}")
                    loader = PyPDFLoader(str(file_path))
                    docs = loader.load()
                elif file_path.suffix.lower() in ['.docx', '.txt']:
                    logger.info(f"Processing document: {file_path}")
                    loader = UnstructuredLoader(str(file_path))
                    docs = loader.load()
                else:
                    logger.warning(f"Unsupported file type: {file_path}")
                    continue

                # Split documents into chunks
                chunks = self.text_splitter.split_documents(docs)

                # Add metadata to each chunk
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        'source_file': str(file_path),
                        'doc_type': self._determine_doc_type(file_path),
                        'date_added': str(Path(file_path).stat().st_mtime),
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    })

                documents.extend(chunks)
                # Record processed file
                self._record_processed_file(str(file_path))
                logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks created")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")

        # Store documents in vector database
        if documents:
            self._store_documents(documents)
        else:
            logger.warning("No new documents to process")

    def _reset_vector_db(self):
        """Delete existing vector database"""
        if self.db_dir.exists():
            logger.warning(f"Deleting existing vector database at {self.db_dir}")
            shutil.rmtree(self.db_dir)
            # Also reset processed files record
            self._clear_processed_files()
        logger.info("Vector database reset")

    def _get_processed_files(self) -> set:
        """Get list of already processed files"""
        processed_files_path = self.db_dir / "processed_files.txt"
        if processed_files_path.exists():
            return set(processed_files_path.read_text().splitlines())
        return set()

    def _record_processed_file(self, file_path: str):
        """Record that a file has been processed"""
        processed_files_path = self.db_dir / "processed_files.txt"
        with processed_files_path.open("a") as f:
            f.write(f"{file_path}\n")

    def _clear_processed_files(self):
        """Clear the record of processed files"""
        processed_files_path = self.db_dir / "processed_files.txt"
        if processed_files_path.exists():
            processed_files_path.unlink()

    def _determine_doc_type(self, file_path: Path) -> str:
        """Determine document type based on filename or content"""
        filename = file_path.name.lower()
        if any(term in filename for term in ['ddq', 'questionnaire']):
            return 'DDQ'
        elif any(term in filename for term in ['lpa', 'agreement']):
            return 'LPA'
        else:
            return 'OTHER'

    def _store_documents(self, documents: List) -> None:
        """Store documents in the vector database"""
        if not documents:
            logger.warning("No documents to store")
            return

        try:
            logger.info(f"Storing {len(documents)} document chunks in vector database")
            vectorstore = Chroma(
                persist_directory=str(self.db_dir),
                embedding_function=self.embeddings
            )

            # Add new documents to existing database
            vectorstore.add_documents(documents)
            logger.info("Successfully stored documents in vector database")

        except Exception as e:
            logger.error(f"Error storing documents in vector database: {str(e)}")
            raise

    def get_database_info(self) -> Dict:
        """Get information about the current vector database"""
        if not self.db_dir.exists():
            return {"exists": False, "document_count": 0, "processed_files": []}

        vectorstore = Chroma(
            persist_directory=str(self.db_dir),
            embedding_function=self.embeddings
        )

        return {
            "exists": True,
            "document_count": vectorstore._collection.count(),
            "processed_files": list(self._get_processed_files())
        }