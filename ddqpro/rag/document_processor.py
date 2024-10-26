from typing import List, Dict, Optional
from pathlib import Path
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
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

    def ingest_documents(self) -> None:
        """Process all documents in the corpus directory"""
        logger.info(f"Processing documents from {self.corpus_dir}")

        documents = []
        for file_path in self.corpus_dir.glob('**/*'):
            try:
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
                logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks created")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")

        # Create or update vector store
        if documents:
            self._store_documents(documents)
        else:
            logger.warning("No documents were successfully processed")

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
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.db_dir)
            )
            vectorstore.persist()
            logger.info("Successfully stored documents in vector database")

        except Exception as e:
            logger.error(f"Error storing documents in vector database: {str(e)}")
            raise