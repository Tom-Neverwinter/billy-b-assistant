from abc import ABC, abstractmethod
from typing import Dict, List

class VoiceProvider(ABC):
    """Abstract base class for all voice synthesis providers"""
    
    @abstractmethod
    async def initialize(self, config: Dict) -> bool:
        """Initialize the voice provider with configuration"""
        pass
    
    @abstractmethod
    async def text_to_speech(self, text: str, voice_params: Dict) -> bytes:
        """Convert text to audio bytes"""
        pass
    
    @abstractmethod
    def get_supported_voices(self) -> List[str]:
        """Get list of available voices"""
        pass