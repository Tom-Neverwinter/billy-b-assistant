from typing import Dict, Any, List
from .base import LLMProvider
from .voice.base_voice import VoiceProvider

# Import LLM providers
from .openai_provider import OpenAIProvider
from .ollama_provider import OllamaProvider
from .kobold_provider import KoboldProvider

# Import voice providers
from .voice.openai_voice import OpenAIVoiceProvider
from .voice.chatterai_voice import ChatterAIVoiceProvider
from .voice.xtt_voice import XTTVoiceProvider

class ProviderFactory:
    """Factory for creating and managing LLM and Voice providers"""
    
    # Registry of available LLM providers
    LLM_PROVIDERS = {
        'openai': OpenAIProvider,
        'ollama': OllamaProvider,
        'kobold': KoboldProvider
    }
    
    # Registry of available voice providers
    VOICE_PROVIDERS = {
        'openai': OpenAIVoiceProvider,
        'chatterai': ChatterAIVoiceProvider,
        'xtt': XTTVoiceProvider
    }
    
    @staticmethod
    async def create_llm_provider(config: Dict[str, Any]) -> LLMProvider:
        """Create and initialize LLM provider"""
        provider_name = config.get('provider', 'openai').lower()
        
        if provider_name not in ProviderFactory.LLM_PROVIDERS:
            available = list(ProviderFactory.LLM_PROVIDERS.keys())
            raise ValueError(f"Unknown LLM provider '{provider_name}'. Available: {available}")
        
        print(f"🔧 Creating LLM provider: {provider_name}")
        
        provider_class = ProviderFactory.LLM_PROVIDERS[provider_name]
        provider = provider_class()
        
        if await provider.initialize(config):
            print(f"✅ LLM provider '{provider_name}' initialized successfully")
            return provider
        else:
            raise ConnectionError(f"Failed to initialize LLM provider '{provider_name}'")
    
    @staticmethod
    async def create_voice_provider(config: Dict[str, Any]) -> VoiceProvider:
        """Create and initialize Voice provider"""
        provider_name = config.get('provider', 'openai').lower()
        
        if provider_name not in ProviderFactory.VOICE_PROVIDERS:
            available = list(ProviderFactory.VOICE_PROVIDERS.keys())
            raise ValueError(f"Unknown Voice provider '{provider_name}'. Available: {available}")
        
        print(f"🔧 Creating Voice provider: {provider_name}")
        
        provider_class = ProviderFactory.VOICE_PROVIDERS[provider_name]
        provider = provider_class()
        
        if await provider.initialize(config):
            print(f"✅ Voice provider '{provider_name}' initialized successfully")
            return provider
        else:
            raise ConnectionError(f"Failed to initialize Voice provider '{provider_name}'")
    
    @staticmethod
    def get_available_llm_providers() -> List[str]:
        """Get list of available LLM providers"""
        return list(ProviderFactory.LLM_PROVIDERS.keys())
    
    @staticmethod  
    def get_available_voice_providers() -> List[str]:
        """Get list of available Voice providers"""
        return list(ProviderFactory.VOICE_PROVIDERS.keys())
    
    @staticmethod
    def validate_llm_config(config: Dict[str, Any]) -> bool:
        """Validate LLM provider configuration"""
        provider_name = config.get('provider', 'openai').lower()
        
        if provider_name == 'openai':
            return bool(config.get('api_key'))
        elif provider_name == 'ollama':
            return bool(config.get('api_url'))
        elif provider_name == 'kobold':
            return bool(config.get('api_url'))
        
        return False
    
    @staticmethod
    def validate_voice_config(config: Dict[str, Any]) -> bool:
        """Validate Voice provider configuration"""
        provider_name = config.get('provider', 'openai').lower()
        
        if provider_name == 'openai':
            return bool(config.get('api_key'))
        elif provider_name in ['chatterai', 'xtt']:
            return bool(config.get('api_url'))
        
        return False