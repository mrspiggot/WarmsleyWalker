# ddqpro/gui/components/sidebar.py

import streamlit as st
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path


@dataclass
class ModelConfig:
    name: str
    description: str
    context_window: int
    suggested_use: str


class Sidebar:
    MODEL_CONFIGS = {
        "OpenAI": {
            "gpt-4o": ModelConfig(
                name="GPT-4o",
                description="Latest GPT-4 'omni' model with 128k context window",
                context_window=128000,
                suggested_use="Smarter model, higher price per token"
            ),
            "gpt-4o-mini": ModelConfig(
                name="GPT-4o-mini",
                description="Mini GPT-4o model with 128k context window",
                context_window=128000,
                suggested_use="GPT-4o mini (“o” for “omni”) is a fast, affordable small model for focused tasks. "
            ),
            "gpt-4-turbo-preview": ModelConfig(
                name="GPT-4 Turbo",
                description="Latest GPT-4 model with 128k context window",
                context_window=128000,
                suggested_use="Best for complex DDQs with large context"
            ),
            "gpt-4": ModelConfig(
                name="GPT-4",
                description="Standard GPT-4 model",
                context_window=8192,
                suggested_use="Good balance of performance and cost"
            ),
            "gpt-3.5-turbo": ModelConfig(
                name="GPT-3.5 Turbo",
                description="Faster, more economical model",
                context_window=16384,
                suggested_use="Best for simpler DDQs or testing"
            )
        },
        "Ollama": {  # Add this block
            "llama3.1:70b": ModelConfig(
                name="Llama 3.1 70b",
                description="A powerful open-source language model.",
                context_window=4096,  # Adjust as needed
                suggested_use="Good for general DDQ processing with an open-source model."
            ),
            "mixtral:8x22b": ModelConfig(
                name="Mixtral 8x22b",
                description="A powerful open-source language model.",
                context_window=4096,  # Adjust as needed
                suggested_use="Good for general DDQ processing with an open-source model."
            ),
            "TiagoG/json-response": ModelConfig(
                name="TiagoG/json-response",
                description="A powerful open-source language model.",
                context_window=4096,  # Adjust as needed
                suggested_use="Good for general DDQ processing with an open-source model."
            ),
            "llama2": ModelConfig(
                name="Llama 2",
                description="A powerful open-source language model.",
                context_window=4096,  # Adjust as needed
                suggested_use="OK for fast inference with an open-source model."
            ),
            # Add other Ollama models as needed
        },

        "Anthropic": {
            "claude-3-opus-20240229": ModelConfig(
                name="Claude 3 Opus",
                description="Most capable Claude model",
                context_window=200000,
                suggested_use="Best for complex analysis and longest context"
            ),
            "claude-3-sonnet-20240229": ModelConfig(
                name="Claude 3 Sonnet",
                description="Balanced performance model",
                context_window=200000,
                suggested_use="Good balance of capability and speed"
            ),
            "claude-3-haiku-20240307": ModelConfig(
                name="Claude 3 Haiku",
                description="Fastest Claude model",
                context_window=200000,
                suggested_use="Best for rapid processing of simple DDQs"
            )
        },
        "Google": {
            "gemini-1.5-pro": ModelConfig(
                name="Gemini 1.5 Pro",
                description="Latest Gemini model",
                context_window=100000,
                suggested_use="Best for complex tasks with long context"
            ),
            "gemini-1.0-pro": ModelConfig(
                name="Gemini 1.0 Pro",
                description="Standard Gemini model",
                context_window=32768,
                suggested_use="Good for general DDQ processing"
            )
        },

    }

    def __init__(self):
        self.assets_path = Path(__file__).parent.parent.parent / 'assets'

    def render(self) -> Tuple[str, str, str]:
        """Render the sidebar and return selected configuration"""
        with st.sidebar:
            st.title("DDQ Processor")

            # Load and display logo if available
            logo_path = self.assets_path / 'logo.png'
            if logo_path.exists():
                st.image(str(logo_path), width=200)

            st.divider()

            # Model Selection Section
            st.header("Model Configuration")

            provider = st.selectbox(
                "Select Provider",
                options=list(self.MODEL_CONFIGS.keys()),
                key='provider_select'
            )

            model_options = list(self.MODEL_CONFIGS[provider].keys())
            model_name = st.selectbox(
                "Select Model",
                options=model_options,
                key='model_select'
            )

            # Display model information
            if model_name:
                model_config = self.MODEL_CONFIGS[provider][model_name]
                with st.expander("Model Details", expanded=False):
                    st.markdown(f"""
                    **{model_config.name}**

                    {model_config.description}

                    * Context Window: {model_config.context_window:,} tokens
                    * Suggested Use: {model_config.suggested_use}
                    """)

            st.divider()

            # Processing Options
            st.header("Processing Options")

            batch_size = st.slider(
                "Batch Size",
                min_value=1,
                max_value=50,
                value=10,
                help="Number of questions to process in parallel"
            )

            max_tokens = st.number_input(
                "Max Response Tokens",
                min_value=100,
                max_value=4000,
                value=1000,
                help="Maximum tokens for each response"
            )

            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="Controls randomness in responses"
            )

            st.divider()

            # Debug Options
            with st.expander("Debug Options", expanded=False):
                debug_mode = st.checkbox(
                    "Enable Debug Mode",
                    help="Show detailed processing information"
                )
                show_costs = st.checkbox(
                    "Show Cost Estimates",
                    help="Display estimated API costs"
                )

            # Create configuration dictionary
            config = {
                "provider": provider,
                "model_name": model_name,
                "batch_size": batch_size,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "debug_mode": debug_mode,
                "show_costs": show_costs
            }

            # Save configuration button
            if st.button("Save Configuration"):
                self._save_config(config)
                st.success("Configuration saved!")

            return provider, model_name, config

    def _save_config(self, config: Dict):
        """Save configuration to a file"""
        config_path = self.assets_path / 'config.json'
        import json

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)