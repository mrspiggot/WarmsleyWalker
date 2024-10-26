from typing import List, Dict
import logging
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class RAGRetriever:
    def __init__(self, db_dir: str = "data/vectordb"):
        self.db_dir = db_dir
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            persist_directory=db_dir,
            embedding_function=self.embeddings
        )

        # Initialize LLM for context compression
        llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview")

        # Create compressor for better context extraction
        compressor = LLMChainExtractor.from_llm(llm)

        # Create retriever with compression
        self.retriever = ContextualCompressionRetriever(
            base_retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 4}
            ),
            doc_compressor=compressor,
        )

    def get_relevant_context(self, question: str, metadata_filter: Dict = None) -> List[Dict]:
        """Retrieve relevant context for a question"""
        try:
            # Get relevant documents
            docs = self.retriever.get_relevant_documents(
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