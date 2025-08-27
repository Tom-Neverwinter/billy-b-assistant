#!/usr/bin/env python3
"""
Billy Bass Assistant - Multi-Provider Edition

A Raspberry Pi-powered voice assistant embedded inside a Big Mouth Billy Bass.
Supports multiple AI providers: OpenAI (GPT-4/5), Ollama, Kobold AI
Supports multiple voice providers: OpenAI TTS, ChatterAI, XTT
"""

import asyncio
import configparser
import os
import sys
import signal
import json
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from personality import (
        PersonalityProfile,
        load_traits_from_ini,
    )
except ImportError:
    print("Warning: personality module not found - using minimal personality system")
    class PersonalityProfile:
        def __init__(self, **kwargs):
            pass
        def generate_prompt(self):
            return "You are Billy, a talking fish with attitude."
    
    def load_traits_from_ini(path):
        return {}

# Import our new provider system
try:
    from providers.factory import ProviderFactory
    from providers.base import ModelType
    PROVIDERS_AVAILABLE = True
    print("✅ Multi-provider system loaded successfully")
except ImportError as e:
    print(f"❌ Provider system not available: {e}")
    print("📝 Make sure you've created the providers/ directory structure")
    PROVIDERS_AVAILABLE = False

# ============================================================================
# CONFIGURATION SYSTEM
# ============================================================================

# === Paths ===
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(ROOT_DIR, ".env")
PERSONA_PATH = os.path.join(ROOT_DIR, "persona.ini")

# === Load .env ===
load_dotenv(dotenv_path=ENV_PATH)

# === Load personality ===
try:
    traits = load_traits_from_ini(PERSONA_PATH)
    PERSONALITY = PersonalityProfile(**traits)
    
    _config = configparser.ConfigParser()
    _config.read(PERSONA_PATH)
    
    EXTRA_INSTRUCTIONS = _config.get("META", "instructions") if _config.has_section("META") else ""
    if _config.has_section("BACKSTORY"):
        BACKSTORY = dict(_config.items("BACKSTORY"))
        BACKSTORY_FACTS = "\n".join([f"- {key}: {value}" for key, value in BACKSTORY.items()])
    else:
        BACKSTORY = {}
        BACKSTORY_FACTS = "You are Billy Bass, a talking fish with no configured backstory."
        
except Exception as e:
    print(f"Warning: Could not load personality configuration: {e}")
    PERSONALITY = PersonalityProfile()
    EXTRA_INSTRUCTIONS = ""
    BACKSTORY_FACTS = "You are Billy Bass, a talking fish with attitude."

# === Instructions for AI ===
BASE_INSTRUCTIONS = """
You also have special powers:
- If someone asks if you like fishsticks you answer Yes. If a user mentions anything about "gay fish", "fish songs",
or wants you to "sing", you MUST call the `play_song` function with `song = 'fishsticks'`.
- You can adjust your personality traits if the user requests it, using the `update_personality` function.
- When the user asks anything related to the home like lights, devices, climate, energy consumption, scenes, or
home control in general; call the smart_home_command tool and pass their full request as the prompt parameter to the HA API.
You will get a response back from Home Assistant itself so you have to interpret and explain it to the end user.

You are allowed to call tools mid-conversation to trigger special behaviors.

DO NOT explain or confirm that you are triggering a tool. Just smoothly integrate it.
"""

INSTRUCTIONS = (
    BASE_INSTRUCTIONS.strip()
    + "\n\n"
    + EXTRA_INSTRUCTIONS.strip()
    + "\n\n"
    + "Known facts about your past:\n"
    + BACKSTORY_FACTS
    + "\n\n"
    + PERSONALITY.generate_prompt()
)

# ============================================================================
# PROVIDER CONFIGURATION
# ============================================================================

@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 150

@dataclass 
class VoiceConfig:
    provider: str
    voice: str
    model: str
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    speed: float = 1.0

@dataclass
class BillyConfig:
    llm: LLMConfig
    voice: VoiceConfig
    # Hardware settings
    billy_model: str = "modern"
    button_pin: int = 27
    mic_timeout: int = 5
    silence_threshold: int = 2000
    # Debug settings
    debug_mode: bool = False
    debug_include_delta: bool = False
    # Integration settings
    mqtt_host: Optional[str] = None
    mqtt_port: int = 1883
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    ha_host: Optional[str] = None
    ha_token: Optional[str] = None
    ha_lang: str = "en"

def detect_llm_provider() -> str:
    """Auto-detect LLM provider based on configuration"""
    if os.getenv("LLM_PROVIDER"):
        return os.getenv("LLM_PROVIDER").lower()
    if os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_URL"):
        return "ollama"
    if os.getenv("KOBOLD_URL"):
        return "kobold"
    return "openai"

def detect_voice_provider() -> str:
    """Auto-detect voice provider based on configuration"""
    if os.getenv("VOICE_PROVIDER"):
        return os.getenv("VOICE_PROVIDER").lower()
    if os.getenv("CHATTERAI_API_KEY"):
        return "chatterai"
    if os.getenv("XTT_API_URL"):
        return "xtt"
    return "openai"

def load_config() -> BillyConfig:
    """Load Billy configuration"""
    # Detect providers
    llm_provider = detect_llm_provider()
    voice_provider = detect_voice_provider()
    
    # LLM Configuration
    if llm_provider == "openai":
        llm_config = LLMConfig(
            provider="openai",
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini-realtime-preview"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_url=os.getenv("OPENAI_API_URL"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "150"))
        )
    elif llm_provider == "ollama":
        api_url = os.getenv("LLM_API_URL") or os.getenv("OLLAMA_URL")
        if not api_url and os.getenv("OLLAMA_HOST"):
            host = os.getenv("OLLAMA_HOST")
            api_url = f"http://{host}" if "://" not in host else host
        llm_config = LLMConfig(
            provider="ollama",
            model=os.getenv("LLM_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2:latest")),
            api_url=api_url or "http://localhost:11434",
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "150"))
        )
    elif llm_provider == "kobold":
        llm_config = LLMConfig(
            provider="kobold",
            model="kobold",
            api_url=os.getenv("LLM_API_URL", os.getenv("KOBOLD_URL", "http://localhost:5000")),
            api_key=os.getenv("KOBOLD_API_KEY"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "150"))
        )
    else:
        llm_config = LLMConfig(provider="openai", model="gpt-4o-mini-realtime-preview")
    
    # Voice Configuration  
    if voice_provider == "openai":
        voice_config = VoiceConfig(
            provider="openai",
            voice=os.getenv("VOICE", "ash"),
            model=os.getenv("OPENAI_VOICE_MODEL", "tts-1"),
            api_key=os.getenv("OPENAI_VOICE_API_KEY", os.getenv("OPENAI_API_KEY")),
            api_url=os.getenv("OPENAI_VOICE_API_URL"),
            speed=float(os.getenv("VOICE_SPEED", "1.0"))
        )
    elif voice_provider == "chatterai":
        voice_config = VoiceConfig(
            provider="chatterai",
            voice=os.getenv("VOICE", "natural"),
            model=os.getenv("CHATTERAI_MODEL", "natural"),
            api_key=os.getenv("CHATTERAI_API_KEY"),
            api_url=os.getenv("VOICE_API_URL", os.getenv("CHATTERAI_URL", "https://api.chatterai.com")),
            speed=float(os.getenv("VOICE_SPEED", "1.0"))
        )
    elif voice_provider == "xtt":
        voice_config = VoiceConfig(
            provider="xtt",
            voice=os.getenv("VOICE", "default"),
            model=os.getenv("XTT_MODEL", "default"),
            api_key=os.getenv("XTT_API_KEY"),
            api_url=os.getenv("VOICE_API_URL", os.getenv("XTT_API_URL", "http://localhost:8080")),
            speed=float(os.getenv("VOICE_SPEED", "1.0"))
        )
    else:
        voice_config = VoiceConfig(provider="openai", voice="ash", model="tts-1")
    
    return BillyConfig(
        llm=llm_config,
        voice=voice_config,
        billy_model=os.getenv("BILLY_MODEL", "modern").lower(),
        button_pin=int(os.getenv("BUTTON_PIN", "27")),
        mic_timeout=int(os.getenv("MIC_TIMEOUT_SECONDS", "5")),
        silence_threshold=int(os.getenv("SILENCE_THRESHOLD", "2000")),
        debug_mode=os.getenv("DEBUG_MODE", "true").lower() == "true",
        debug_include_delta=os.getenv("DEBUG_MODE_INCLUDE_DELTA", "false").lower() == "true",
        mqtt_host=os.getenv("MQTT_HOST") or None,
        mqtt_port=int(os.getenv("MQTT_PORT", "1883")),
        mqtt_username=os.getenv("MQTT_USERNAME") or None,
        mqtt_password=os.getenv("MQTT_PASSWORD") or None,
        ha_host=os.getenv("HA_HOST"),
        ha_token=os.getenv("HA_TOKEN"),
        ha_lang=os.getenv("HA_LANG", "en").lower()
    )

# Load configuration
CONFIG = load_config()

# Legacy variables for backward compatibility
OPENAI_API_KEY = CONFIG.llm.api_key if CONFIG.llm.provider == "openai" else os.getenv("OPENAI_API_KEY", "")
VOICE = CONFIG.voice.voice
DEBUG_MODE = CONFIG.debug_mode
BILLY_MODEL = CONFIG.billy_model

def is_classic_billy():
    return CONFIG.billy_model == "classic"

# ============================================================================
# BILLY BASS ASSISTANT CLASS
# ============================================================================

class BillyBassAssistant:
    """Main Billy Bass Assistant with multi-provider support"""
    
    def __init__(self):
        self.config = CONFIG
        self.llm_provider = None
        self.voice_provider = None
        self.session_active = False
        self.shutdown_requested = False
        
        # Hardware components (to be implemented based on existing Billy setup)
        self.motor_controller = None
        self.audio_player = None
        self.button_handler = None
        
        print(f"🐟 Billy Bass Assistant initializing...")
        print(f"📡 LLM Provider: {self.config.llm.provider}")
        print(f"🔊 Voice Provider: {self.config.voice.provider}")
    
    async def initialize(self):
        """Initialize Billy Bass with configured providers"""
        if not PROVIDERS_AVAILABLE:
            print("❌ Provider system not available - check providers/ directory")
            return False
        
        try:
            # Initialize LLM Provider
            print(f"🔧 Initializing {self.config.llm.provider} LLM provider...")
            llm_config = {
                'provider': self.config.llm.provider,
                'model': self.config.llm.model,
                'api_key': self.config.llm.api_key,
                'api_url': self.config.llm.api_url,
                'temperature': self.config.llm.temperature,
                'max_tokens': self.config.llm.max_tokens
            }
            self.llm_provider = await ProviderFactory.create_llm_provider(llm_config)
            
            # Initialize Voice Provider
            print(f"🔧 Initializing {self.config.voice.provider} voice provider...")
            voice_config = {
                'provider': self.config.voice.provider,
                'voice': self.config.voice.voice,
                'model': self.config.voice.model,
                'api_key': self.config.voice.api_key,
                'api_url': self.config.voice.api_url,
                'speed': self.config.voice.speed
            }
            self.voice_provider = await ProviderFactory.create_voice_provider(voice_config)
            
            # Initialize hardware components
            await self.initialize_hardware()
            
            print(f"✅ Billy Bass initialized successfully!")
            print(f"🎯 Model: {self.config.llm.model}")
            print(f"🔊 Voice: {self.config.voice.voice}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Billy Bass: {e}")
            return False
    
    async def initialize_hardware(self):
        """Initialize Billy Bass hardware components"""
        print("🔧 Initializing hardware components...")
        
        # TODO: Initialize based on existing Billy Bass hardware code:
        # - GPIO setup for motors
        # - Audio system setup
        # - Button interrupt setup
        # - Motor controller setup
        
        print("🎛️  Hardware initialization complete")
    
    async def start_conversation(self):
        """Start a new conversation session"""
        if not self.llm_provider:
            print("❌ No LLM provider available")
            return False
        
        try:
            await self.llm_provider.start_session()
            self.session_active = True
            print("🗣️  Conversation session started")
            return True
        except Exception as e:
            print(f"❌ Failed to start conversation: {e}")
            return False
    
    async def process_user_input(self, user_message: str):
        """Process user input and generate response"""
        if not self.session_active:
            if not await self.start_conversation():
                return
        
        try:
            print(f"👤 User: {user_message}")
            
            # Send message to LLM provider
            await self.llm_provider.send_message(user_message)
            
            # Handle response based on provider type
            model_type = self.llm_provider.get_model_type()
            
            if model_type == ModelType.REALTIME:
                await self._handle_realtime_response()
            elif model_type == ModelType.STREAMING:
                await self._handle_streaming_response()
            else:
                await self._handle_complete_response()
                
        except Exception as e:
            print(f"❌ Error processing user input: {e}")
    
    async def _handle_realtime_response(self):
        """Handle realtime streaming response (OpenAI style)"""
        print("🤖 Billy (realtime):")
        # TODO: Integrate existing OpenAI Realtime API handling here
        pass
    
    async def _handle_streaming_response(self):
        """Handle streaming response (Ollama style)"""
        complete_text = ""
        print("🤖 Billy: ", end="", flush=True)
        
        try:
            async for chunk in self.llm_provider.get_response_stream():
                if chunk["type"] == "text_delta":
                    text = chunk["text"]
                    print(text, end="", flush=True)
                    complete_text += text
                elif chunk["type"] == "error":
                    print(f"\n❌ Error: {chunk['message']}")
                    return
                elif chunk.get("done", False):
                    print()  # New line
                    break
            
            # Generate voice response
            if complete_text.strip():
                await self._generate_voice_response(complete_text.strip())
                
        except Exception as e:
            print(f"\n❌ Error in streaming response: {e}")
    
    async def _handle_complete_response(self):
        """Handle complete response (Kobold style)"""
        try:
            async for response in self.llm_provider.get_response_stream():
                if response["type"] == "text_complete":
                    text = response["text"]
                    if text.strip():
                        print(f"🤖 Billy: {text}")
                        await self._generate_voice_response(text)
                elif response["type"] == "error":
                    print(f"❌ Error: {response['message']}")
                    
        except Exception as e:
            print(f"❌ Error in complete response: {e}")
    
    async def _generate_voice_response(self, text: str):
        """Generate and play voice response"""
        if not self.voice_provider:
            print("⚠️  No voice provider available")
            return
        
        try:
            print("🔊 Generating voice...")
            
            voice_params = {
                'voice': self.config.voice.voice,
                'speed': self.config.voice.speed
            }
            
            audio_data = await self.voice_provider.text_to_speech(text, voice_params)
            
            # TODO: Integrate with existing Billy Bass audio playback and motor control
            await self._play_audio_with_animation(audio_data, text)
            
        except Exception as e:
            print(f"❌ Voice generation failed: {e}")
    
    async def _play_audio_with_animation(self, audio_data: bytes, text: str):
        """Play audio with Billy Bass motor animation"""
        print(f"🎵 Playing audio ({len(audio_data)} bytes)...")
        
        # TODO: Integrate existing Billy Bass functionality:
        # - Motor control for mouth movement
        # - Tail movement
        # - Head movement
        # - Audio playback through speakers
        # - Timing synchronization
        
        print("🎭 Animation complete")
    
    async def handle_button_press(self):
        """Handle physical button press"""
        print("🔘 Button pressed - starting voice session...")
        
        # TODO: Integrate existing button handling and voice recording
        # This would typically:
        # 1. Record audio from microphone
        # 2. Convert speech to text (or use voice mode)
        # 3. Process the input
        
        # For now, simulate with text input
        user_input = input("💬 What would you like to say to Billy? ")
        if user_input.strip():
            await self.process_user_input(user_input)
    
    async def shutdown(self):
        """Gracefully shutdown Billy Bass"""
        print("🐟 Shutting down Billy Bass...")
        
        self.shutdown_requested = True
        
        if self.llm_provider:
            await self.llm_provider.end_session()
        
        # TODO: Cleanup hardware components
        # - Stop motor controllers
        # - Release GPIO pins
        # - Close audio devices
        
        print("👋 Billy Bass shutdown complete")
    
    async def run(self):
        """Main run loop"""
        print("🚀 Billy Bass Assistant starting...")
        
        if not await self.initialize():
            print("❌ Failed to initialize Billy Bass")
            return
        
        print("🎤 Billy Bass is ready! Press button to start conversation.")
        print("💡 For testing, you can also type messages directly.")
        
        try:
            while not self.shutdown_requested:
                # TODO: Replace with actual button/voice detection
                # For now, use simple input for testing
                try:
                    user_input = input("\n💬 Say something to Billy (or 'quit' to exit): ")
                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        break
                    elif user_input.strip():
                        await self.process_user_input(user_input)
                except KeyboardInterrupt:
                    break
                except EOFError:
                    break
                    
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
        finally:
            await self.shutdown()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_system_info():
    """Print system information and configuration"""
    print("=" * 60)
    print("🐟 BILLY BASS ASSISTANT - MULTI-PROVIDER EDITION")
    print("=" * 60)
    print(f"📡 LLM Provider: {CONFIG.llm.provider}")
    print(f"🤖 Model: {CONFIG.llm.model}")
    print(f"🔊 Voice Provider: {CONFIG.voice.provider}")
    print(f"🎵 Voice: {CONFIG.voice.voice}")
    print(f"🎛️  Billy Model: {CONFIG.billy_model}")
    print(f"🔧 Debug Mode: {CONFIG.debug_mode}")
    
    if CONFIG.mqtt_host:
        print(f"📡 MQTT: {CONFIG.mqtt_host}:{CONFIG.mqtt_port}")
    if CONFIG.ha_host:
        print(f"🏠 Home Assistant: {CONFIG.ha_host}")
    
    if PROVIDERS_AVAILABLE:
        available_llm = ProviderFactory.get_available_llm_providers()
        available_voice = ProviderFactory.get_available_voice_providers()
        print(f"🔌 Available LLM: {available_llm}")
        print(f"🔌 Available Voice: {available_voice}")
    
    print("=" * 60)

def setup_signal_handlers(billy):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}")
        billy.shutdown_requested = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point"""
    print_system_info()
    
    # Create Billy Bass instance
    billy = BillyBassAssistant()
    
    # Setup signal handlers
    setup_signal_handlers(billy)
    
    # Run Billy Bass
    await billy.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Billy Bass Assistant terminated")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)
