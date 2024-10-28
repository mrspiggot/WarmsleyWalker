from typing import List, Dict
import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

logger = logging.getLogger(__name__)


class RAGRetriever:
    def __init__(self, db_dir: str = "data/vectordb"):
        self.db_dir = db_dir
        self.embeddings = OpenAIEmbeddings()

        # Initialize the base retriever
        self.vectorstore = Chroma(
            persist_directory=db_dir,
            embedding_function=self.embeddings
        )
        self.base_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        # Cache for section contexts
        self._section_context_cache = {}

    async def aget_section_context(self, section_name: str) -> List[Dict]:
        """Get context relevant to an entire section"""

        # Check cache first
        if section_name in self._section_context_cache:
            logger.debug(f"Using cached context for section {section_name}")
            return self._section_context_cache[section_name]

        try:
            # Construct section-specific query
            section_query = f"information about {section_name} in DDQ responses"

            # Get documents from vector store
            docs = await self.base_retriever.aget_relevant_documents(
                section_query,
                filter={"doc_type": "DDQ"}
            )

            # Format results
            results = []
            for doc in docs:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source': doc.metadata.get('source_file', 'Unknown')
                })

            # Cache the results
            self._section_context_cache[section_name] = results

            logger.info(f"Retrieved {len(results)} context chunks for section {section_name}")
            return results

        except Exception as e:
            logger.error(f"Error retrieving section context: {str(e)}")
            # Return empty list rather than raising to allow graceful degradation
            return []

    async def aget_relevant_context(self, question: str, metadata_filter: Dict = None) -> List[Dict]:
        """Retrieve relevant context for a specific question"""
        try:
            # Get relevant documents using base retriever
            docs = await self.base_retriever.aget_relevant_documents(
                question,
                filter=metadata_filter
            )

            # Format results
            results = []
            for doc in docs:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source': doc.metadata.get('source_file', 'Unknown')
                })

            logger.info(f"Retrieved {len(results)} relevant context chunks for question")
            return results

        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            # Return empty list rather than raising to allow graceful degradation
            return []

    def get_relevant_context(self, question: str, metadata_filter: Dict = None) -> List[Dict]:
        """Synchronous version of context retrieval for backward compatibility"""
        try:
            # Get relevant documents using base retriever
            docs = self.base_retriever.get_relevant_documents(
                question,
                filter=metadata_filter
            )

            # Format results
            results = []
            for doc in docs:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source': doc.metadata.get('source_file', 'Unknown')
                })

            logger.info(f"Retrieved {len(results)} relevant context chunks for question")
            return results

        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            raise


