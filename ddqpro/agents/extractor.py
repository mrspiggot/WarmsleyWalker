# ddqpro/agents/extractor.py
from ddqpro.models.state import DDQState
from ddqpro.tools.document_tools import PDFExtractor, DocxExtractor

class DocumentExtractor:
    def extract(self, state: DDQState) -> DDQState:
        print(f"Extracting content from: {state['input_path']}")
        # Just pass through for now
        return state