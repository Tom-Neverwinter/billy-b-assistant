#!/usr/bin/env python3
"""
Test script for Billy Bass providers

Usage:
    python test_providers.py ollama
    python test_providers.py kobold  
    python test_providers.py chatterai
    python test_providers.py openai
"""

import asyncio
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from providers.factory import ProviderFactory

async def test_ollama():
    """Test Ollama provider"""
    print("🔧 Testing Ollama provider...")
    
    config = {
        'provider': 'ollama',
        'api_url': 'http://localhost:11434',  # Change to your Ollama server
        'model': 'llama3.2:latest',
        'temperature': 0.7,
        'max_tokens': 100
    }
    
    try:
        provider = await ProviderFactory.create_llm_provider(config)
        await provider.start_session()
        await provider.send_message("Hello! Tell me a short joke about fish.")
        
        print("🤖 Billy's response:")
        complete_response = ""
        async for chunk in provider.get_response_stream():
            if chunk["type"] == "text_delta":
                print(chunk["text"], end="", flush=True)
                complete_response += chunk["text"]
            elif chunk["type"] == "error":
                print(f"\n❌ Error: {chunk['message']}")
            elif chunk.get("done"):
                print("\n✅ Ollama test complete!")
                break
                
        await provider.end_session()
        
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")

async def test_kobold():
    """Test Kobold provider"""
    print("🔧 Testing Kobold provider...")
    
    config = {
        'provider': 'kobold',
        'api_url': 'http://localhost:5000',  # Change to your Kobold server
        'temperature': 0.7,
        'max_tokens': 100
    }
    
    try:
        provider = await ProviderFactory.create_llm_provider(config)
        await provider.start_session()
        await provider.send_message("Hello! Tell me a short joke about fish.")
        
        print("🤖 Billy's response:")
        async for chunk in provider.get_response_stream():
            if chunk["type"] == "text_complete":
                print(chunk["text"])
                print("✅ Kobold test complete!")
            elif chunk["type"] == "error":
                print(f"❌ Error: {chunk['message']}")
                
        await provider.end_session()
        
    except Exception as e:
        print(f"❌ Kobold test failed: {e}")

async def test_chatterai():
    """Test ChatterAI voice provider"""
    print("🔧 Testing ChatterAI voice provider...")
    
    config = {
        'provider': 'chatterai',
        'api_url': 'http://localhost:8080',  # Change to your ChatterAI server
        'api_key': 'your-api-key-here',      # Add your API key
        'model': 'natural'
    }
    
    try:
        provider = await ProviderFactory.create_voice_provider(config)
        
        audio_data = await provider.text_to_speech(
            "Hello! I'm Billy Bass, and I'm testing my new ChatterAI voice!",
            {"voice": "natural", "speed": 1.0}
        )
        
        # Save test audio
        with open('test_chatterai.wav', 'wb') as f:
            f.write(audio_data)
        
        print("✅ ChatterAI test complete! Audio saved to test_chatterai.wav")
        
    except Exception as e:
        print(f"❌ ChatterAI test failed: {e}")

async def test_openai():
    """Test OpenAI providers"""
    print("🔧 Testing OpenAI providers...")
    
    # Test LLM
    llm_config = {
        'provider': 'openai',
        'api_key': 'sk-your-key-here',  # Add your OpenAI API key
        'model': 'gpt-4o-mini-realtime-preview'
    }
    
    # Test Voice  
    voice_config = {
        'provider': 'openai',
        'api_key': 'sk-your-key-here',  # Add your OpenAI API key
        'model': 'tts-1'
    }
    
    try:
        # Test LLM
        print("Testing OpenAI LLM...")
        llm_provider = await ProviderFactory.create_llm_provider(llm_config)
        print("✅ OpenAI LLM provider initialized")
        
        # Test Voice
        print("Testing OpenAI Voice...")
        voice_provider = await ProviderFactory.create_voice_provider(voice_config)
        
        audio_data = await voice_provider.text_to_speech(
            "Hello! I'm Billy Bass with OpenAI voice!",
            {"voice": "ash", "speed": 1.0}
        )
        
        with open('test_openai.wav', 'wb') as f:
            f.write(audio_data)
        
        print("✅ OpenAI test complete! Audio saved to test_openai.wav")
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")

async def main():
    """Main test function"""
    if len(sys.argv) != 2:
        print("Usage: python test_providers.py <provider>")
        print("Providers: ollama, kobold, chatterai, openai")
        return
    
    provider = sys.argv[1].lower()
    
    if provider == 'ollama':
        await test_ollama()
    elif provider == 'kobold':
        await test_kobold()
    elif provider == 'chatterai':
        await test_chatterai()
    elif provider == 'openai':
        await test_openai()
    else:
        print(f"❌ Unknown provider: {provider}")
        print("Available: ollama, kobold, chatterai, openai")

if __name__ == "__main__":
    asyncio.run(main())