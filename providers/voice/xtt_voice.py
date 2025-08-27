import aiohttp
from typing import Dict, List
from .base_voice import VoiceProvider

class XTTVoiceProvider(VoiceProvider):
    """Client-only XTT provider - connects to external XTT/Coqui server"""
    
    def __init__(self):
        self.api_url = None
        self.api_key = None
        self.model = "default"
    
    async def initialize(self, config: Dict) -> bool:
        """Initialize connection to external XTT server"""
        self.api_url = config.get('api_url', 'http://localhost:8080')
        self.api_key = config.get('api_key')  # Optional for local XTT
        self.model = config.get('model', 'default')
        
        # Test connection to XTT server
        try:
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            async with aiohttp.ClientSession() as session:
                # Test if XTT server is running
                async with session.get(
                    f"{self.api_url}/api/v1/voices",
                    headers=headers
                ) as resp:
                    if resp.status == 200:
                        voices_data = await resp.json()
                        available_voices = voices_data.get('voices', ['default'])
                        print(f"✅ Connected to XTT server at {self.api_url}")
                        print(f"🔊 Available voices: {available_voices}")
                        return True
                    else:
                        print(f"❌ XTT server returned status {resp.status}")
                        return False
                        
        except Exception as e:
            print(f"❌ Failed to connect to XTT server at {self.api_url}: {e}")
            return False
    
    async def text_to_speech(self, text: str, voice_params: Dict) -> bytes:
        """Convert text to speech using external XTT server"""
        headers = {'Content-Type': 'application/json'}
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
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
                        print(f"🔊 XTT generated {len(audio_data)} bytes of audio")
                        return audio_data
                    else:
                        error_text = await resp.text()
                        raise Exception(f"XTT TTS error {resp.status}: {error_text}")
                        
        except Exception as e:
            print(f"❌ XTT TTS failed: {e}")
            raise e
    
    def get_supported_voices(self) -> List[str]:
        """Get list of supported XTT voices"""
        return [
            "default",
            "female",
            "male",
            "robotic",
            "natural",
            "expressive"
        ]