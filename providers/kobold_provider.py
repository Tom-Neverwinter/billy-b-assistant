import aiohttp
from typing import Dict, Any, AsyncGenerator
from .base import LLMProvider, ModelType

class KoboldProvider(LLMProvider):
    """Client-only Kobold provider - connects to external Kobold server"""
    
    def __init__(self):
        self.api_url = None
        self.temperature = 0.7
        self.max_tokens = 150
        self.max_context = 2048
        self.conversation_history = []
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize connection to external Kobold server"""
        self.api_url = config.get('api_url', 'http://localhost:5000')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 150)
        self.max_context = config.get('max_context', 2048)
        
        # Test connection to external Kobold server
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/api/v1/model") as resp:
                    if resp.status == 200:
                        model_info = await resp.json()
                        model_name = model_info.get('result', 'Unknown Model')
                        print(f"✅ Connected to Kobold server at {self.api_url}")
                        print(f"🤖 Loaded model: {model_name}")
                        return True
                    else:
                        print(f"❌ Kobold server returned status {resp.status}")
                        return False
        except Exception as e:
            print(f"❌ Failed to connect to Kobold server at {self.api_url}: {e}")
            return False
    
    async def start_session(self) -> bool:
        """Start new conversation session"""
        self.conversation_history = []
        print(f"🐟 Billy started new Kobold session")
        return True
    
    async def send_message(self, message: str) -> None:
        """Add user message to conversation"""
        self.conversation_history.append(f"User: {message}")
        print(f"👤 User: {message}")
    
    async def get_response_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Get response from external Kobold server"""
        # Format conversation for Kobold (simple text format)
        prompt = "\n".join(self.conversation_history) + "\nAssistant:"
        
        # Truncate if too long for context window
        if len(prompt) > self.max_context:
            # Keep recent conversation history
            lines = prompt.split('\n')
            while len('\n'.join(lines)) > self.max_context and len(lines) > 2:
                lines.pop(0)
            prompt = '\n'.join(lines)
        
        payload = {
            "prompt": prompt,
            "max_length": self.max_tokens,
            "temperature": self.temperature,
            "rep_pen": 1.1,
            "top_p": 0.9,
            "top_k": 40,
            "sampler_order": [6, 0, 1, 2, 3, 4, 5]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/api/v1/generate",
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        response_text = data.get('results', [{}])[0].get('text', '').strip()
                        
                        if response_text:
                            # Clean up the response (remove any prompt echoing)
                            if response_text.startswith('Assistant:'):
                                response_text = response_text[10:].strip()
                            
                            # Add to conversation history
                            self.conversation_history.append(f"Assistant: {response_text}")
                            print(f"🤖 Billy: {response_text}")
                        
                        # Kobold returns complete response (not streaming)
                        yield {
                            "type": "text_complete", 
                            "text": response_text,
                            "done": True
                        }
                        
                    else:
                        error_text = await resp.text()
                        yield {
                            "type": "error", 
                            "message": f"Kobold server error {resp.status}: {error_text}"
                        }
                        
        except Exception as e:
            yield {
                "type": "error", 
                "message": f"Connection error to Kobold server: {e}"
            }
    
    async def end_session(self) -> None:
        """End the current session"""
        print(f"🐟 Billy ended Kobold session")
        pass
    
    def get_model_type(self) -> ModelType:
        return ModelType.REQUEST_RESPONSE