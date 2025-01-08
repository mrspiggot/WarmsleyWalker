from typing import List, Dict
import logging
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    def __init__(self, db_dir: str = "data/vectordb"):
        self.db_dir = db_dir
        self.embeddings = OpenAIEmbeddings()

        # Instead of Chroma, use InMemoryVectorStore
        self.vectorstore = InMemoryVectorStore(embedding=self.embeddings)
        self.base_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        # Cache for section contexts
        self._section_context_cache = {}

    async def aget_section_context(self, section_name: str) -> List[Dict]:
        # example usage with a filter, or remove if not needed
        query = f"information about {section_name}"
        # “filter”: Optional usage. If you want doc_type == "DDQ" only
        docs = await self.vectorstore.asimilarity_search(
            query=query,
            k=4,
            filter=lambda d: d.metadata.get("doc_type") == "DDQ"
        )
        # Format results
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source_file", "Unknown")
            }
            for doc in docs
        ]

    async def aget_relevant_context(self, question: str, metadata_filter: Dict = None) -> List[Dict]:
        """Retrieve relevant context for a specific question"""
        try:
            # Get relevant documents using base retriever
            # docs = await self.base_retriever.aget_relevant_documents(
            #     question,
            #     filter=metadata_filter
            # )
            docs = await self.base_retriever.ainvoke(
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


