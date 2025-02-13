import logging
from typing import List, Dict, Optional, Callable
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
logger = logging.getLogger(__name__)




class RAGRetriever:
    def __init__(self, vector_store=None, db_dir: str = "data/vectordb"):
        self.db_dir = db_dir
        self.embeddings = OpenAIEmbeddings()
        if vector_store is not None:
            self.vectorstore = vector_store
        else:
            self.vectorstore = InMemoryVectorStore(embedding=self.embeddings)
        self.base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        self._section_context_cache = {}


    async def debug_print_documents(self):
        """Print information about available documents"""
        try:
            # Get all documents from the vector store using similarity search with a dummy query.
            results = await self.vectorstore.asimilarity_search(query="test", k=999)
            print(f"\nTotal documents in vectorstore: {len(results)}")
            print("\nFirst few documents:")
            for i, doc in enumerate(results[:3]):
                print(f"\nDocument {i + 1}:")
                print(f"Content preview: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}")
        except Exception as e:
            print(f"Error printing documents: {str(e)}")

    async def aget_section_context(self, section_name: str) -> List[Dict]:
        """Get section-level context by searching for documents related to the section."""
        query = f"information about {section_name}"
        docs = await self.vectorstore.asimilarity_search(
            query=query,
            k=4,
            filter=lambda d: d.get("metadata", {}).get("doc_type") == "DDQ"
        )
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source_file", "Unknown")
            }
            for doc in docs
        ]

    async def aget_relevant_context(self, question: str, metadata_filter: Optional[Dict] = None) -> List[Dict]:
        """Retrieve relevant context for a specific question."""
        try:
            logger.info(f"Retrieving context for question: {question[:100]}...")
            logger.debug(f"Using metadata filter: {metadata_filter}")
            print(f"\nSearching for context: {question[:100]}...")

            if isinstance(metadata_filter, dict):
                def filter_func(d):
                    meta = getattr(d, "metadata", {}) or {}
                    return all(meta.get(key) == value for key, value in metadata_filter.items())
            else:
                filter_func = metadata_filter

            docs = await self.vectorstore.asimilarity_search(
                query=question,
                k=4,
                filter=filter_func
            )
            print(f"Found {len(docs)} relevant documents")
            for i, doc in enumerate(docs):
                print(f"Doc {i}: type={type(doc)}")
                # Use getattr to safely access metadata
                meta = getattr(doc, "metadata", {})
                print(f"Metadata: {meta}")

            results = []
            for doc in docs:
                try:
                    meta = getattr(doc, "metadata", {}) or {}
                    result = {
                        "content": getattr(doc, "page_content", str(doc)),
                        "metadata": meta,
                        "source": meta.get("source_file", "Unknown")
                    }
                    results.append(result)
                    print(f"Added document: {result['source']}")
                    print(f"Content preview: {result['content'][:200]}...")
                except Exception as e:
                    logger.error(f"Error formatting document: {str(e)}")
                    continue
            return results

        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
