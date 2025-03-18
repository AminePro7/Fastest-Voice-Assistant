#!/usr/bin/env python3
"""
Voice-only assistant with automatic speech recognition.
Uses the speech_recognition library for speech-to-text.
Automatically listens every 5 seconds and detects language.
"""

import sys
import os
import asyncio
import time
import speech_recognition as sr
from datetime import datetime
from langdetect import detect as detect_language

# Add the current directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from voice_assistant.nlp.language_model import language_model
from voice_assistant.speech.tts.edge_tts_generator import generate_speech
from voice_assistant.config.settings import ASSISTANT_SETTINGS

# Language mapping from langdetect to speech_recognition
LANGUAGE_MAPPING = {
    'en': 'en-US',
    'fr': 'fr-FR',
    'es': 'es-ES',
    'de': 'de-DE',
    'it': 'it-IT',
    'pt': 'pt-BR',
    'ru': 'ru-RU',
    'ja': 'ja-JP',
    'ko': 'ko-KR',
    'zh-cn': 'zh-CN',
    'zh-tw': 'zh-TW',
    'ar': 'ar-AE',
    'nl': 'nl-NL',
    'pl': 'pl-PL',
    'tr': 'tr-TR'
}

class AutoSpeechRecognition:
    """Automatic speech recognition with language detection."""
    
    def __init__(self):
        """Initialize speech recognition."""
        # Create recognizer
        self.recognizer = sr.Recognizer()
        
        # Set a low energy threshold for better sensitivity
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = False
        
        # Default language
        self.language = "en-US"
        
        # Find available microphones
        self.list_microphones()
        
        # Try to find the best microphone
        self.mic_index = self.find_best_microphone()
        
        # Flag to control continuous listening
        self.running = True
    
    def list_microphones(self):
        """List available microphones."""
        try:
            mic_list = sr.Microphone.list_microphone_names()
            print("\nAvailable microphones:")
            for i, name in enumerate(mic_list):
                print(f"  {i}: {name}")
            return mic_list
        except Exception as e:
            print(f"Error listing microphones: {e}")
            return []
    
    def find_best_microphone(self):
        """Find the best microphone to use."""
        try:
            mic_list = sr.Microphone.list_microphone_names()
            
            # Look for headset or microphone in the name
            for i, name in enumerate(mic_list):
                if "headset" in name.lower() or "microphone" in name.lower():
                    print(f"\nSelected microphone {i}: {name}")
                    return i
            
            # If no good microphone found, use default (index 0)
            if mic_list:
                print(f"\nUsing default microphone: {mic_list[0]}")
                return 0
            
            return None
        except Exception as e:
            print(f"Error finding microphone: {e}")
            return None
    
    def listen(self):
        """Listen for speech and convert to text."""
        print(f"\nListening for 5 seconds...")
        
        try:
            # Create microphone instance
            if self.mic_index is not None:
                microphone = sr.Microphone(device_index=self.mic_index)
            else:
                microphone = sr.Microphone()
            
            with microphone as source:
                # Adjust for ambient noise
                print("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Force a low threshold
                self.recognizer.energy_threshold = 300
                print(f"Energy threshold: {self.recognizer.energy_threshold}")
                
                # Listen for speech
                print("Speak now...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                
                # Try to recognize speech
                try:
                    print("Recognizing...")
                    # First try without language specification for auto-detection
                    text = self.recognizer.recognize_google(audio)
                    print(f"Recognized: {text}")
                    
                    # Detect language from the recognized text
                    try:
                        detected_lang = detect_language(text)
                        full_lang_code = LANGUAGE_MAPPING.get(detected_lang, 'en-US')
                        print(f"Detected language: {detected_lang} ({full_lang_code})")
                        self.language = full_lang_code
                        
                        # If language is not English, try to recognize again with the detected language
                        if detected_lang != 'en':
                            text = self.recognizer.recognize_google(audio, language=self.language)
                            print(f"Re-recognized with {self.language}: {text}")
                    except Exception as lang_error:
                        print(f"Language detection error: {lang_error}")
                    
                    return text
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    print(f"Recognition error: {e}")
        
        except Exception as e:
            print(f"Error: {e}")
        
        return None
    
    def set_microphone(self, index):
        """Set the microphone to use."""
        try:
            mic_list = sr.Microphone.list_microphone_names()
            if 0 <= index < len(mic_list):
                self.mic_index = index
                print(f"Set microphone to {index}: {mic_list[index]}")
                return True
            else:
                print(f"Invalid microphone index: {index}")
                return False
        except Exception as e:
            print(f"Error setting microphone: {e}")
            return False
    
    async def continuous_listen(self, callback):
        """Continuously listen for speech and call the callback with recognized text."""
        while self.running:
            text = self.listen()
            if text:
                await callback(text)
            
            # Wait a moment before listening again
            print("\nWaiting 2 seconds before listening again...")
            await asyncio.sleep(2)

async def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("       VOICE-ONLY ASSISTANT")
    print("=" * 60)
    print("\nA voice-only assistant that automatically listens every 5 seconds.")
    print("Language is automatically detected from your speech.")
    print("\nCommands:")
    print("  - Press Ctrl+C to exit")
    print("=" * 60 + "\n")
    
    # Initialize language model
    print("Setting up language model...")
    language_model.setup(ASSISTANT_SETTINGS["system_prompt"])
    print("Language model initialized!")
    
    # Initialize speech recognition
    speech = AutoSpeechRecognition()
    
    # Welcome message
    welcome = "Welcome to the Voice-Only Assistant. I will automatically listen for your voice every 5 seconds and detect the language you're speaking. You don't need to type anything."
    print(f"\nAssistant: {welcome}")
    
    # Try to generate speech
    try:
        await generate_speech(welcome)
    except Exception as e:
        print(f"Error generating speech: {e}")
    
    # Conversation history
    conversation = []
    
    # Callback for when speech is recognized
    async def on_speech_recognized(text):
        # Add to conversation history
        conversation.append({"role": "user", "content": text})
        
        # Get response from language model
        print("Thinking...")
        response = language_model.generate_response(text, context=conversation[-10:])
        
        if response:
            print(f"\nAssistant: {response}")
            
            # Add to conversation history
            conversation.append({"role": "assistant", "content": response})
            
            # Generate speech
            try:
                await generate_speech(response)
            except Exception as e:
                print(f"Error generating speech: {e}")
        else:
            print("\nAssistant: I'm not sure how to respond to that.")
    
    # Start continuous listening
    try:
        await speech.continuous_listen(on_speech_recognized)
    except KeyboardInterrupt:
        print("\nExiting...")
        speech.running = False
    except Exception as e:
        print(f"Error: {e}")
        speech.running = False

if __name__ == "__main__":
    asyncio.run(main()) 