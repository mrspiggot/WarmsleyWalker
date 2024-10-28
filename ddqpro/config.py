from langchain.globals import set_debug
from langchain.globals import set_verbose
import logging


def setup_async_env(debug: bool = False):
    """Setup environment for async operations"""
    # Configure LangChain
    set_debug(debug)
    set_verbose(debug)

    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure specific loggers
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('langchain').setLevel(log_level)