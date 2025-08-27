import asyncio
import configparser
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

from .personality import (
    PersonalityProfile,
    load_traits_from_ini,
)

# === Paths ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT_DIR, ".env")
PERSONA_PATH = os.path.join(ROOT_DIR, "persona.ini")

# === Load .env ===
load_dotenv(dotenv_path=ENV_PATH)

# === Load traits.ini ===
traits = load_traits_from_ini(PERSONA_PATH)

# === Build Personality ===
PERSONALITY = PersonalityProfile(**traits)

_config = configparser.ConfigParser()
_config.read(PERSONA_PATH)

# === Instructions for GPT ===
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

EXTRA_INSTRUCTIONS = _config.get("META", "instructions")
if _config.has_section("BACKSTORY"):
    BACKSTORY = dict(_config.items("BACKSTORY"))
    BACKSTORY_FACTS = "\n".join([
        f"- {key}: {value}" for key, value in BACKSTORY.items()
    ])
else:
    BACKSTORY = {}
    BACKSTORY_FACTS = (
        "You are an enigma and nobody knows anything about you because the person "
        "talking to you hasn't configured your backstory. You might remind them to do "
        "that."
    )

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

# === OpenAI Config (EXISTING - UNCHANGED) ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini-realtime-preview")
VOICE = os.getenv("VOICE", "ash")

# === Modes ===
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"
DEBUG_MODE_INCLUDE_DELTA = (
    os.getenv("DEBUG_MODE_INCLUDE_DELTA", "false").lower() == "true"
)
TEXT_ONLY_MODE = os.getenv("TEXT_ONLY_MODE", "false").lower() == "true"
RUN_MODE = os.getenv("RUN_MODE", "normal").lower()

# === Billy Hardware ===
BILLY_MODEL = os.getenv("BILLY_MODEL", "modern").strip().lower()

# === Audio Config ===
SPEAKER_PREFERENCE = os.getenv("SPEAKER_PREFERENCE")
MIC_PREFERENCE = os.getenv("MIC_PREFERENCE")
MIC_TIMEOUT_SECONDS = int(os.getenv("MIC_TIMEOUT_SECONDS", "5"))
SILENCE_THRESHOLD = int(os.getenv("SILENCE_THRESHOLD", "2000"))
CHUNK_MS = int(os.getenv("CHUNK_MS", "50"))
PLAYBACK_VOLUME = 1

# === GPIO Config ===
BUTTON_PIN = int(os.getenv("BUTTON_PIN", "27"))

# === MQTT Config (EXISTING - UNCHANGED) ===
MQTT_HOST = os.getenv("MQTT_HOST", "")
MQTT_PORT = int(os.getenv("MQTT_PORT", "0"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")

# === Home Assistant Config (EXISTING - UNCHANGED) ===
HA_HOST = os.getenv("HA_HOST")
HA_TOKEN = os.getenv("HA_TOKEN")
HA_LANG = os.getenv("HA_LANG", "en")

# === Personality Config ===
ALLOW_UPDATE_PERSONALITY_INI = (
    os.getenv("ALLOW_UPDATE_PERSONALITY_INI", "true").lower() == "true"
)

# === Software Config ===
FLASK_PORT = int(os.getenv("FLASK_PORT", "80"))


def is_classic_billy():
    return os.getenv("BILLY_MODEL", "modern").strip().lower() == "classic"


try:
    MAIN_LOOP = asyncio.get_event_loop()
except RuntimeError:
    MAIN_LOOP = None


# ============================================================================
# NEW MULTI-PROVIDER SYSTEM (BACKWARD COMPATIBLE)
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
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    speed: float = 1.0

@dataclass
class MQTTConfig:
    host: Optional[str] = None
    port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None

@dataclass
class HomeAssistantConfig:
    host: Optional[str] = None
    token: Optional[str] = None
    language: str = "en"

@dataclass
class BillyConfig:
    llm: LLMConfig
    voice: VoiceConfig
    mqtt: MQTTConfig
    home_assistant: HomeAssistantConfig
    # Existing settings preserved
    mic_timeout: int = 5
    silence_threshold: int = 900
    debug_mode: bool = False
    debug_include_delta: bool = False
    allow_personality_updates: bool = True
    billy_model: str = "modern"
    flask_port: int = 80
    button_pin: int = 27


def detect_llm_provider() -> str:
    """Auto-detect LLM provider based on available configuration"""
    # Check for explicit provider override
    if os.getenv("LLM_PROVIDER"):
        return os.getenv("LLM_PROVIDER").lower()
    
    # Check for Ollama settings
    if os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_URL") or os.getenv("LLM_API_URL"):
        return "ollama"
    
    # Check for Kobold settings  
    if os.getenv("KOBOLD_URL") or os.getenv("KOBOLD_HOST"):
        return "kobold"
    
    # Default to OpenAI (backward compatibility)
    return "openai"


def detect_voice_provider() -> str:
    """Auto-detect voice provider based on available configuration"""
    # Check for explicit provider override
    if os.getenv("VOICE_PROVIDER"):
        return os.getenv("VOICE_PROVIDER").lower()
    
    # Check for ChatterAI settings
    if os.getenv("CHATTERAI_API_KEY") or (os.getenv("VOICE_API_URL") and "chatter" in os.getenv("VOICE_API_URL", "").lower()):
        return "chatterai"
    
    # Check for XTT settings
    if os.getenv("XTT_API_URL") or os.getenv("XTT_HOST") or (os.getenv("VOICE_API_URL") and "xtt" in os.getenv("VOICE_API_URL", "").lower()):
        return "xtt"
    
    # Default to OpenAI (backward compatibility)
    return "openai"


def get_llm_model(provider: str) -> str:
    """Get model name based on provider"""
    if provider == "openai":
        return OPENAI_MODEL  # Use existing variable
    elif provider == "ollama":
        return os.getenv("LLM_MODEL") or os.getenv("OLLAMA_MODEL", "llama3.2:latest")
    elif provider == "kobold":
        return os.getenv("LLM_MODEL", "kobold")
    return OPENAI_MODEL  # Fallback to existing


def get_llm_api_key(provider: str) -> Optional[str]:
    """Get API key based on provider"""
    if provider == "openai":
        return OPENAI_API_KEY  # Use existing variable
    elif provider == "ollama":
        return os.getenv("OLLAMA_API_KEY")  # Usually None for local
    elif provider == "kobold":
        return os.getenv("KOBOLD_API_KEY")
    return OPENAI_API_KEY  # Fallback to existing


def get_llm_api_url(provider: str) -> Optional[str]:
    """Get API URL based on provider"""
    if provider == "openai":
        return None  # Use OpenAI's default
    elif provider == "ollama":
        # Support multiple env var names for flexibility
        if os.getenv("LLM_API_URL"):
            return os.getenv("LLM_API_URL")
        elif os.getenv("OLLAMA_URL"):
            return os.getenv("OLLAMA_URL")
        elif os.getenv("OLLAMA_HOST"):
            host = os.getenv("OLLAMA_HOST")
            if "://" not in host:
                host = f"http://{host}"
            return host
        return "http://localhost:11434"  # Default
    elif provider == "kobold":
        return os.getenv("LLM_API_URL") or os.getenv("KOBOLD_URL", "http://localhost:5000")
    return None


def get_voice_api_key(provider: str) -> Optional[str]:
    """Get voice API key based on provider"""
    if provider == "openai":
        return OPENAI_API_KEY  # Reuse existing key
    elif provider == "chatterai":
        return os.getenv("CHATTERAI_API_KEY")
    elif provider == "xtt":
        return os.getenv("XTT_API_KEY")
    return OPENAI_API_KEY  # Fallback to existing


def get_voice_api_url(provider: str) -> Optional[str]:
    """Get voice API URL based on provider"""
    if provider == "openai":
        return None  # Use OpenAI's default
    elif provider == "chatterai":
        return os.getenv("VOICE_API_URL") or os.getenv("CHATTERAI_URL", "https://api.chatterai.com")
    elif provider == "xtt":
        return os.getenv("VOICE_API_URL") or os.getenv("XTT_API_URL") or os.getenv("XTT_HOST", "http://localhost:8080")
    return None


def load_billy_config() -> BillyConfig:
    """Load complete Billy configuration with backward compatibility"""
    
    # Detect providers
    llm_provider = detect_llm_provider()
    voice_provider = detect_voice_provider()
    
    # LLM Configuration
    llm_config = LLMConfig(
        provider=llm_provider,
        model=get_llm_model(llm_provider),
        api_key=get_llm_api_key(llm_provider),
        api_url=get_llm_api_url(llm_provider),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "150"))
    )
    
    # Voice Configuration  
    voice_config = VoiceConfig(
        provider=voice_provider,
        voice=VOICE,  # Use existing VOICE variable
        api_key=get_voice_api_key(voice_provider),
        api_url=get_voice_api_url(voice_provider),
        speed=float(os.getenv("VOICE_SPEED", "1.0"))
    )
    
    # MQTT Configuration (use existing variables)
    mqtt_config = MQTTConfig(
        host=MQTT_HOST if MQTT_HOST else None,
        port=MQTT_PORT if MQTT_PORT > 0 else 1883,
        username=MQTT_USERNAME if MQTT_USERNAME else None,
        password=MQTT_PASSWORD if MQTT_PASSWORD else None
    )
    
    # Home Assistant Configuration (use existing variables)  
    ha_config = HomeAssistantConfig(
        host=HA_HOST,
        token=HA_TOKEN,
        language=HA_LANG.lower()
    )
    
    return BillyConfig(
        llm=llm_config,
        voice=voice_config,
        mqtt=mqtt_config,
        home_assistant=ha_config,
        # Use existing variables
        mic_timeout=MIC_TIMEOUT_SECONDS,
        silence_threshold=SILENCE_THRESHOLD,
        debug_mode=DEBUG_MODE,
        debug_include_delta=DEBUG_MODE_INCLUDE_DELTA,
        allow_personality_updates=ALLOW_UPDATE_PERSONALITY_INI,
        billy_model=BILLY_MODEL,
        flask_port=FLASK_PORT,
        button_pin=BUTTON_PIN
    )


# ============================================================================
# NEW PROVIDER-SPECIFIC CONFIGURATIONS
# ============================================================================

# === Ollama Provider Config ===
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
OLLAMA_URL = os.getenv("OLLAMA_URL", f"http://{OLLAMA_HOST}")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")  # Usually None

# === Kobold AI Provider Config ===
KOBOLD_URL = os.getenv("KOBOLD_URL", "http://localhost:5000")
KOBOLD_HOST = os.getenv("KOBOLD_HOST", "localhost:5000")
KOBOLD_API_KEY = os.getenv("KOBOLD_API_KEY")
KOBOLD_MAX_CONTEXT = int(os.getenv("KOBOLD_MAX_CONTEXT", "2048"))

# === ChatterAI Provider Config ===
CHATTERAI_API_KEY = os.getenv("CHATTERAI_API_KEY")
CHATTERAI_URL = os.getenv("CHATTERAI_URL", "https://api.chatterai.com")

# === XTT Provider Config ===
XTT_API_KEY = os.getenv("XTT_API_KEY")
XTT_API_URL = os.getenv("XTT_API_URL", "http://localhost:8080")
XTT_HOST = os.getenv("XTT_HOST", "localhost:8080")

# === Generic Provider Overrides ===
LLM_PROVIDER = os.getenv("LLM_PROVIDER", detect_llm_provider())
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_API_URL = os.getenv("LLM_API_URL")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "150"))

VOICE_PROVIDER = os.getenv("VOICE_PROVIDER", detect_voice_provider())
VOICE_API_URL = os.getenv("VOICE_API_URL")
VOICE_SPEED = float(os.getenv("VOICE_SPEED", "1.0"))


# ============================================================================
# CONVENIENCE FUNCTIONS FOR EXISTING CODE
# ============================================================================

def get_current_llm_provider() -> str:
    """Get currently configured LLM provider"""
    return detect_llm_provider()


def get_current_voice_provider() -> str:
    """Get currently configured voice provider"""
    return detect_voice_provider()


def is_using_openai_llm() -> bool:
    """Check if using OpenAI for LLM (backward compatibility)"""
    return get_current_llm_provider() == "openai"


def is_using_openai_voice() -> bool:
    """Check if using OpenAI for voice (backward compatibility)"""
    return get_current_voice_provider() == "openai"


def get_effective_api_key() -> str:
    """Get the API key for current LLM provider (backward compatibility)"""
    provider = get_current_llm_provider()
    return get_llm_api_key(provider) or ""


# ============================================================================
# BACKWARD COMPATIBILITY GLOBALS
# ============================================================================

# These ensure existing code continues to work without changes
BILLY_CONFIG = load_billy_config()  # Structured config for new code
CURRENT_LLM_PROVIDER = get_current_llm_provider()  # For existing code
CURRENT_VOICE_PROVIDER = get_current_voice_provider()  # For existing code
