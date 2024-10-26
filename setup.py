from setuptools import setup, find_packages

setup(
    name="ddqpro",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain-openai",
        "langgraph",
        "pymupdf4llm",
        "python-dotenv",
        "pydantic",
        "typing-extensions"
    ],
    entry_points={
        'console_scripts': [
            'ddqpro=ddqpro.main:main',
        ],
    },
    author="Warmsley Walker",
    description="AI-powered DDQ processing tool",
    python_requires=">=3.9",
)