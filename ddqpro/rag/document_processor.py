from typing import List, Dict, Optional
from pathlib import Path
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel
import shutil
from langchain_unstructured import UnstructuredLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
import pdfplumber

logger = logging.getLogger(__name__)
class PDFPlumberLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        docs = []
        with pdfplumber.open(self.file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                metadata = {"source_file": self.file_path, "page_number": i}
                docs.append(Document(page_content=text, metadata=metadata))
        return docs


class DocumentMetadata(BaseModel):
    """Metadata for processed documents"""
    source_file: str
    doc_type: str  # DDQ, LPA, etc.
    date_added: str
    chunk_index: int
    total_chunks: int


class CorpusProcessor:
    def __init__(self, corpus_dir: str = "data/corpus"):
        self.corpus_dir = Path(corpus_dir)
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        # InMemoryVectorStore will store all docs in memory only.
        # If you want to persist them across sessions, this won't do that.
        self.vector_store = InMemoryVectorStore(embedding=self.embeddings)

    def ingest_documents(self, reset_db: bool = False) -> None:
        """Process all documents in the corpus directory and store them in memory."""
        logger.info(f"\n=== Starting Document Ingestion ===")
        logger.info(f"Corpus directory: {self.corpus_dir}")

        # If resetting, clear existing documents
        if reset_db:
            logger.info("Resetting document store...")
            self.vector_store = InMemoryVectorStore(embedding=self.embeddings)

        documents = []

        # Traverse all files in corpus dir
        for file_path in self.corpus_dir.glob("**/*"):
            if not file_path.is_file():
                continue

            try:
                logger.info(f"\nProcessing file: {file_path}")
                if file_path.suffix.lower() == ".pdf":
                    loader = PDFPlumberLoader(str(file_path))
                    docs = loader.load()
                    logger.info(f"Loaded {len(docs)} pages from PDF")
                elif file_path.suffix.lower() in [".docx", ".txt"]:
                    loader = UnstructuredLoader(str(file_path))
                    docs = loader.load()
                    logger.info(f"Loaded document: {len(docs)} sections")
                else:
                    logger.warning(f"Skipping unsupported file type: {file_path}")
                    continue

                # Split into chunks
                chunks = self.text_splitter.split_documents(docs)
                logger.info(f"Split into {len(chunks)} chunks")

                # Add metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "source_file": str(file_path),
                        "doc_type": self._determine_doc_type(file_path),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    })

                documents.extend(chunks)
                logger.info(f"Processed {file_path}: {len(chunks)} chunks added")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")

        # Add all documents to vector store
        if documents:
            logger.info(f"\nAdding {len(documents)} chunks to vector store...")
            self.vector_store.add_documents(documents=documents)
            logger.info("Documents added successfully")
        else:
            logger.warning("No documents processed")


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
            return 'DDQ'
        else:
            return 'DDQ'

    def _store_documents(self, documents: List) -> None:
        if not documents:
            logger.warning("No documents to store")
            return

        try:
            logger.info(f"Storing {len(documents)} document chunks in an in-memory vector store")
            # 1) Create an in-memory store
            vectorstore = InMemoryVectorStore(embedding=self.embeddings)

            # 2) Add documents
            vectorstore.add_documents(documents=documents)
            logger.info("Successfully stored documents in memory (non-persistent)")

            # Because it’s in-memory only, you lose the ability to load these docs
            # later once the Python process exits.

        except Exception as e:
            logger.error(f"Error storing documents in in-memory vector store: {str(e)}")
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