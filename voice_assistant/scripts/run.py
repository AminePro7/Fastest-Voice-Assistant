#!/usr/bin/env python3
"""
Run script for the Multilingual Voice Assistant.

This script provides a command-line interface to run the voice assistant.
"""

import asyncio
import os
import sys
import tempfile

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from voice_assistant.core.assistant import VoiceAssistant
from voice_assistant.config.settings import PATHS
from voice_assistant.speech.tts.edge_tts_generator import cleanup as tts_cleanup

def print_header():
    """Print the header."""
    print("\n" + "=" * 60)
    print("       MULTILINGUAL VOICE ASSISTANT")
    print("=" * 60)
    print("\nA terminal-based voice assistant with real-time speech recognition")
    print("and natural text-to-speech powered by Gemini AI.")
    print("Supports multiple languages for input and output.")
    print("\nFeatures:")
    print("  ✅ Multilingual speech recognition and response")
    print("  ✅ Enhanced voice activity detection")
    print("  ✅ Automatic language detection")
    print("  ✅ Natural text-to-speech in the detected language")
    print("=" * 60 + "\n")

async def cleanup_resources():
    """Clean up resources when exiting."""
    print("Cleaning up resources...")
    await tts_cleanup()
    print("Cleanup complete.")

async def main():
    """Main function."""
    print_header()
    
    # Set temp directory in settings
    PATHS["temp_dir"] = tempfile.gettempdir()
    
    # Create the voice assistant
    assistant = VoiceAssistant()
    
    # Run the assistant
    try:
        # Start the assistant and block until it's done
        await assistant.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        assistant.stop()
        await cleanup_resources()

def run_assistant():
    """Run the assistant in the current thread."""
    try:
        # Get or create an event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the main function and block until complete
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nExiting voice assistant...")
    finally:
        # Clean up the loop
        loop.close()

if __name__ == "__main__":
    run_assistant() 