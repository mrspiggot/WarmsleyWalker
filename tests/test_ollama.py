from pathlib import Path
import json
import logging
from ddqpro.models.llm_manager import LLMManager
from ddqpro.agents.analyzer import DocumentAnalyzer
from ddqpro.models.state import DDQState

# Basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
)
logger = logging.getLogger(__name__)


class DDQQuestionTest:
    def __init__(self):
        self.models = [
            "mixtral:8x22b",
            "llama3.1:70b",
            "TiagoG/json-response"
        ]
        self.analyzer = DocumentAnalyzer()

        # Get project root and set paths
        self.project_root = Path(__file__).parent.parent
        self.ddq_file = self.project_root / "data" / "sample" / "AIMA DDQ.pdf"
        self.output_dir = self.project_root / "data" / "output"
        self.output_dir.mkdir(exist_ok=True)

    def test_model(self, model_name: str):
        """Test a single model's question extraction"""
        logger.info(f"Testing {model_name}")

        try:
            # Initialize model
            llm = LLMManager()
            llm.initialize("ollama", model_name)

            # Create basic state
            state = DDQState(
                input_path=str(self.ddq_file),
                file_type='.pdf',
                current_extractor="default",
                extraction_results=None,
                reflections=[],
                json_output=None,
                response_states={},
                completed_responses={},
                cost_tracking={}
            )

            # Extract questions
            result = self.analyzer.analyze(state)

            # Save results
            if result.get('extraction_results'):
                output = {
                    'model': model_name,
                    'sections': result['extraction_results'].content['sections'],
                    'questions': result['extraction_results'].content['questions']
                }

                output_file = self.output_dir / f"questions_{model_name.replace('/', '_')}.json"
                with open(output_file, 'w') as f:
                    json.dump(output, f, indent=2)

                logger.info(f"Saved results for {model_name}")
            else:
                logger.error(f"{model_name} failed to extract questions")

        except Exception as e:
            logger.error(f"{model_name} error: {str(e)}")

    def run_tests(self):
        """Test all models"""
        for model in self.models:
            self.test_model(model)


if __name__ == "__main__":
    tester = DDQQuestionTest()
    tester.run_tests()