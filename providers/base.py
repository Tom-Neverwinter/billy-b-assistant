from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator
from enum import Enum

class ModelType(Enum):
    REALTIME = "realtime"
    STREAMING = "streaming" 
    REQUEST_RESPONSE = "request_response"

class LLMProvider(ABC):
    """Abstract base class for all LLM providers"""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the provider with configuration"""
        pass
    
    @abstractmethod
    async def start_session(self) -> bool:
        """Start a new conversation session"""
        pass
    
    @abstractmethod
    async def send_message(self, message: str) -> None:
        """Send user message to the model"""
        pass
    
    @abstractmethod
    async def get_response_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Get streaming response from model"""
        pass
    
    @abstractmethod
    async def end_session(self) -> None:
        """End the current session"""
        pass
    
    @abstractmethod
    def get_model_type(self) -> ModelType:
        """Return the type of model interaction"""
        pass