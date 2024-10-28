from typing import List, Dict
import logging
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_chroma import Chroma

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

    def get_relevant_context(self, question: str, metadata_filter: Dict = None) -> List[Dict]:
        """Retrieve relevant context for a question"""
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