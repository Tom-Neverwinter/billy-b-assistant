"""
Billy Bass Multi-Provider System

This package provides support for multiple AI and voice providers:

LLM Providers:
- OpenAI (GPT-4, GPT-5, Realtime API)
- Ollama (Local Llama models via server)
- Kobold AI (via external server)

Voice Providers:
- OpenAI (TTS-1, TTS-1-HD)
- ChatterAI (via external server)
- XTT/Coqui (via external server)
"""

from .factory import ProviderFactory
from .base import LLMProvider, ModelType
from .voice.base_voice import VoiceProvider

__all__ = [
    'ProviderFactory',
    'LLMProvider', 
    'VoiceProvider',
    'ModelType'
]