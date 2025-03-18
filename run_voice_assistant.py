#!/usr/bin/env python3
"""
Simple script to run the voice assistant.
"""

import sys
import os
import signal

# Add the current directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from voice_assistant.scripts.run import run_assistant

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully exit"""
    print("\nReceived signal to exit. Cleaning up...")
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Starting voice assistant. Press Ctrl+C to exit.")
    print("The assistant will continue running in this terminal window.")
    print("You can speak to it at any time.")
    
    # Run the assistant in blocking mode
    run_assistant() 