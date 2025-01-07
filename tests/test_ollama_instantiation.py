# test_ollama_initialization.py
from ddqpro.models.llm_factory import LLMFactory
from ddqpro.models.llm_manager import LLMManager
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_llm_initialization():
    logger.debug("\n=== Testing LLM Initialization ===")

    try:
        # Create LLM Factory
        logger.debug("1. Creating LLM Factory")
        factory = LLMFactory()
        logger.debug(f"Available providers: {list(factory.providers.keys())}")

        # Test provider and model name
        provider = "ollama"
        model_name = "llama3.1:70b"
        logger.debug(f"2. Testing with provider: {provider}, model: {model_name}")

        # Initialize LLM Manager (singleton)
        logger.debug("3. Getting LLM Manager instance")
        llm_manager = LLMManager()

        # Initialize the LLM
        logger.debug("4. Initializing LLM")
        llm_manager.initialize(provider, model_name)

        # Verify LLM is initialized
        logger.debug("5. Checking LLM initialization")
        llm = llm_manager.llm
        logger.debug(f"LLM type: {type(llm)}")

        # Test with simple prompt
        logger.debug("6. Testing LLM with simple prompt")
        test_prompt = "Say hello!"
        response = llm.invoke(test_prompt)
        logger.debug(f"Response type: {type(response)}")
        logger.debug(f"Response content: {response}")

        logger.debug("Test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Error during test: {str(e)}", exc_info=True)
        logger.error(f"Error type: {type(e)}")
        return False


if __name__ == "__main__":
    test_llm_initialization()