"""
Text-to-Speech (TTS) implementation using edge-tts.

This module provides TTS functionality for multiple languages using Microsoft Edge TTS.
"""

import asyncio
import edge_tts
from langdetect import detect
import os
import pygame
import tempfile
import uuid
import time

from voice_assistant.config.settings import TTS_SETTINGS

class EdgeTTSGenerator:
    """Text-to-Speech generator using Microsoft Edge TTS."""
    
    def __init__(self, voice_mapping=None):
        """
        Initialize the TTS generator.
        
        Args:
            voice_mapping: Dictionary mapping language codes to voice names.
        """
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        # Use provided voice mapping or default from settings
        self.voice_mapping = voice_mapping or TTS_SETTINGS["voice_mapping"]
        
        # Keep track of temporary files to clean up later
        self.temp_files = []
        
        # Track last message to prevent duplicates
        self.last_message = ""
        self.last_message_time = 0
        self.duplicate_threshold = 2.0  # seconds
    
    async def generate_and_play_speech(self, text):
        """
        Generate and play speech from text.
        
        Args:
            text: Text to convert to speech.
        """
        # Check for duplicate message
        current_time = time.time()
        if (text == self.last_message and 
            current_time - self.last_message_time < self.duplicate_threshold):
            print(f"[TTS] Skipping duplicate message: '{text[:30]}...' (if longer than 30 chars)")
            return
        
        # Update last message tracking
        self.last_message = text
        self.last_message_time = current_time
        
        temp_file = None
        try:
            # Detect language
            detected_lang = detect(text)
            voice = self.voice_mapping.get(detected_lang, 'en-US-ChristopherNeural')
            
            print(f"Detected language: {detected_lang}")
            print(f"[TTS] Generating speech for: '{text[:50]}...' (if longer than 50 chars)")
            
            # Create communicate object
            communicate = edge_tts.Communicate(text, voice)
            
            # Create a unique temporary file with a timestamp to avoid conflicts
            temp_dir = tempfile.gettempdir()
            timestamp = int(time.time() * 1000)
            unique_id = str(uuid.uuid4())
            temp_file = os.path.join(temp_dir, f"tts_output_{timestamp}_{unique_id}.mp3")
            
            # Save audio to temporary file
            await communicate.save(temp_file)
            print(f"[TTS] Audio saved to temporary file: {temp_file}")
            
            # Add to list of temp files to clean up
            self.temp_files.append(temp_file)
            
            # Check if the file exists and has content
            if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
                print(f"[TTS] Error: Generated audio file is empty or doesn't exist")
                return
            
            # Try multiple playback methods
            played = False
            
            # Method 1: Try pygame
            if not played:
                try:
                    # Verify pygame mixer is initialized
                    if not pygame.mixer.get_init():
                        print("[TTS] Initializing pygame mixer...")
                        pygame.mixer.init()
                    
                    print("[TTS] Loading audio file with pygame...")
                    pygame.mixer.music.load(temp_file)
                    print("[TTS] Playing audio with pygame...")
                    pygame.mixer.music.play()
                    
                    # Wait for the audio to finish playing
                    print("[TTS] Waiting for audio to finish...")
                    while pygame.mixer.music.get_busy():
                        await asyncio.sleep(0.1)
                        
                    # Stop playback
                    pygame.mixer.music.stop()
                    print("[TTS] Audio playback complete with pygame")
                    played = True
                except Exception as pygame_error:
                    print(f"[TTS] Error playing audio with pygame: {pygame_error}")
            
            # Method 2: Try system default player
            if not played:
                try:
                    print("[TTS] Trying system default audio player...")
                    if os.name == 'nt':  # Windows
                        os.system(f'start {temp_file}')
                        print("[TTS] Started audio with Windows default player")
                        # Wait a bit for the player to start
                        await asyncio.sleep(2)
                        played = True
                    elif os.name == 'posix':  # Linux/Mac
                        os.system(f'xdg-open {temp_file}')
                        print("[TTS] Started audio with Linux/Mac default player")
                        # Wait a bit for the player to start
                        await asyncio.sleep(2)
                        played = True
                except Exception as system_error:
                    print(f"[TTS] Error playing audio with system player: {system_error}")
            
            # Method 3: Try direct Windows API (Windows only)
            if not played and os.name == 'nt':
                try:
                    print("[TTS] Trying Windows API for audio playback...")
                    import winsound
                    winsound.PlaySound(temp_file, winsound.SND_FILENAME)
                    print("[TTS] Played audio with Windows API")
                    played = True
                except Exception as winsound_error:
                    print(f"[TTS] Error playing audio with Windows API: {winsound_error}")
            
            # If all methods failed, at least make a beep
            if not played:
                try:
                    print("[TTS] All audio playback methods failed. Trying system beep...")
                    if os.name == 'nt':  # Windows
                        import winsound
                        winsound.Beep(1000, 500)  # 1000 Hz for 500 ms
                        print("[TTS] Made a system beep")
                    else:
                        print("\a")  # ASCII bell character
                        print("[TTS] Sent ASCII bell character")
                except Exception as beep_error:
                    print(f"[TTS] Error making system beep: {beep_error}")
            
            # Wait a moment before attempting to clean up
            await asyncio.sleep(0.5)
            
        except Exception as e:
            print(f"[TTS] Error generating speech: {e}")
        finally:
            # Don't delete the file immediately to allow alternative playback methods to work
            pass
    
    async def cleanup(self):
        """Clean up any remaining temporary files."""
        for temp_file in list(self.temp_files):
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    self.temp_files.remove(temp_file)
            except Exception as e:
                print(f"Could not remove temporary file during cleanup: {e}")

# Singleton instance for easy access
tts_generator = EdgeTTSGenerator()

# Convenience function for generating speech
async def generate_speech(text):
    """Generate and play speech from text."""
    await tts_generator.generate_and_play_speech(text)

# Convenience function for cleanup
async def cleanup():
    """Clean up temporary files."""
    await tts_generator.cleanup() 