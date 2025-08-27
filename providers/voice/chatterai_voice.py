import aiohttp
from typing import Dict, List
from .base_voice import VoiceProvider

class ChatterAIVoiceProvider(VoiceProvider):
    """Client-only ChatterAI provider - connects to external ChatterAI server"""
    
    def __init__(self):
        self.api_url = None
        self.api_key = None
        self.model = "natural"
    
    async def initialize(self, config: Dict) -> bool:
        """Initialize connection to external ChatterAI server"""
        self.api_url = config.get('api_url', 'https://api.chatterai.com')
        self.api_key = config.get('api_key')
        self.model = config.get('model', 'natural')
        
        if not self.api_key:
            print("❌ ChatterAI API key not provided")
            return False
        
        # Test connection to ChatterAI server
        try:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/api/v1/voices", 
                    headers=headers
                ) as resp:
                    if resp.status == 200:
                        voices_data = await resp.json()
                        available_voices = voices_data.get('voices', [])
                        print(f"✅ Connected to ChatterAI server at {self.api_url}")
                        print(f"🔊 Available voices: {available_voices}")
                        return True
                    else:
                        print(f"❌ ChatterAI server returned status {resp.status}")
                        return False
        except Exception as e:
            print(f"❌ Failed to connect to ChatterAI server at {self.api_url}: {e}")
            return False
    
    async def text_to_speech(self, text: str, voice_params: Dict) -> bytes:
        """Convert text to speech using external ChatterAI server"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'text': text,
            'voice': voice_params.get('voice', self.model),
            'speed': voice_params.get('speed', 1.0),
            'format': 'wav'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/api/v1/tts",
                    json=payload,
                    headers=headers
                ) as resp:
                    if resp.status == 200:
                        audio_data = await resp.read()
                        print(f"🔊 ChatterAI generated {len(audio_data)} bytes of audio")
                        return audio_data
                    else:
                        error_text = await resp.text()
                        raise Exception(f"ChatterAI TTS error {resp.status}: {error_text}")
                        
        except Exception as e:
            print(f"❌ ChatterAI TTS failed: {e}")
            raise e
    
    def get_supported_voices(self) -> List[str]:
        """Get list of supported ChatterAI voices"""
        return [
            "natural",
            "expressive", 
            "calm",
            "energetic",
            "professional",
            "friendly"
        ]