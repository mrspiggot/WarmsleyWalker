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
        """
        Process all documents in the corpus directory and store them in memory.
        If reset_db=True, we can optionally clear out anything we've previously stored.
        """
        logger.debug("\n=== Starting Document Ingestion ===")
        logger.debug(f"Corpus directory: {self.corpus_dir}")

        if reset_db:
            logger.info("reset_db=True requested; clearing in‑memory documents (if any)")
            # Because InMemoryVectorStore doesn’t persist data, a simple re‑instantiate is enough:
            self.vector_store = InMemoryVectorStore(embedding=self.embeddings)

        logger.info(f"Processing documents from {self.corpus_dir}")

        # Collect new documents as we parse them
        documents = []

        # Traverse all files in corpus dir
        for file_path in self.corpus_dir.glob("**/*"):
            # skip directories
            if not file_path.is_file():
                continue

            try:
                if file_path.suffix.lower() == ".pdf":
                    logger.info(f"Processing PDF: {file_path}")
                    # loader = PyPDFLoader(str(file_path)) Fitz style
                    loader = PDFPlumberLoader(str(file_path))
                    docs = loader.load()
                elif file_path.suffix.lower() in [".docx", ".txt"]:
                    logger.info(f"Processing document: {file_path}")
                    loader = UnstructuredLoader(str(file_path))
                    docs = loader.load()
                else:
                    logger.warning(f"Skipping unsupported file type: {file_path}")
                    continue

                # Split into chunks
                chunks = self.text_splitter.split_documents(docs)

                # Add helpful metadata to each chunk
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "source_file": str(file_path),
                        "doc_type": self._determine_doc_type(file_path),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    })

                # Accumulate them all
                documents.extend(chunks)
                logger.info(
                    f"Successfully processed {file_path}: {len(chunks)} chunks created"
                )

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")

        # Add all the new documents to our in‑memory vector store
        if documents:
            logger.info(f"Adding {len(documents)} document chunks to in‑memory store...")
            self.vector_store.add_documents(documents=documents)
            logger.info("In‑memory ingestion complete.")
        else:
            logger.warning("No new documents found or processed.")


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