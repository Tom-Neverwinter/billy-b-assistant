import aiohttp
from typing import Dict, List
from .base_voice import VoiceProvider

class OpenAIVoiceProvider(VoiceProvider):
    """OpenAI voice provider - maintains backward compatibility with existing Billy setup"""
    
    def __init__(self):
        self.api_key = None
        self.api_url = None
        self.model = "tts-1"
        self.organization = None
        self.project_id = None
    
    async def initialize(self, config: Dict) -> bool:
        """Initialize OpenAI voice connection"""
        self.api_key = config.get('api_key')
        self.api_url = config.get('api_url', 'https://api.openai.com/v1')
        self.model = config.get('model', 'tts-1')
        self.organization = config.get('organization')
        self.project_id = config.get('project_id')
        
        if not self.api_key:
            print("❌ OpenAI API key not provided for voice")
            return False
        
        # Test OpenAI API access
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            if self.organization:
                headers['OpenAI-Organization'] = self.organization
            if self.project_id:
                headers['OpenAI-Project'] = self.project_id
            
            # Simple API test (list models endpoint)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/models",
                    headers=headers
                ) as resp:
                    if resp.status == 200:
                        print(f"✅ OpenAI voice provider initialized with model: {self.model}")
                        return True
                    else:
                        print(f"❌ OpenAI API returned status {resp.status}")
                        return False
                        
        except Exception as e:
            print(f"❌ Failed to connect to OpenAI API: {e}")
            return False
    
    async def text_to_speech(self, text: str, voice_params: Dict) -> bytes:
        """Convert text to speech using OpenAI TTS API"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        if self.organization:
            headers['OpenAI-Organization'] = self.organization
        if self.project_id:
            headers['OpenAI-Project'] = self.project_id
        
        payload = {
            'model': self.model,
            'input': text,
            'voice': voice_params.get('voice', 'ash'),
            'speed': voice_params.get('speed', 1.0),
            'response_format': 'wav'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/audio/speech",
                    json=payload,
                    headers=headers
                ) as resp:
                    if resp.status == 200:
                        audio_data = await resp.read()
                        print(f"🔊 OpenAI generated {len(audio_data)} bytes of audio")
                        return audio_data
                    else:
                        error_text = await resp.text()
                        raise Exception(f"OpenAI TTS error {resp.status}: {error_text}")
                        
        except Exception as e:
            print(f"❌ OpenAI TTS failed: {e}")
            raise e
    
    def get_supported_voices(self) -> List[str]:
        """Get list of supported OpenAI voices"""
        return [
            "alloy",
            "echo", 
            "fable",
            "onyx",
            "nova",
            "shimmer",
            "ash",      # New voices
            "ballad",
            "coral",
            "sage",
            "verse"
        ]