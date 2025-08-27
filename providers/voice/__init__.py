"""
Billy Bass Voice Providers

Voice synthesis providers for Billy Bass:
- OpenAI: TTS-1, TTS-1-HD with latest voices
- ChatterAI: External ChatterAI server
- XTT: External XTT/Coqui server
"""

from .base_voice import VoiceProvider
from .openai_voice import OpenAIVoiceProvider
from .chatterai_voice import ChatterAIVoiceProvider
from .xtt_voice import XTTVoiceProvider

__all__ = [
    'VoiceProvider',
    'OpenAIVoiceProvider',
    'ChatterAIVoiceProvider', 
    'XTTVoiceProvider'
]