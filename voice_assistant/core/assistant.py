"""
Main voice assistant implementation.

This module provides the core voice assistant functionality.
"""

import asyncio
import threading
import queue
import time
import re
from datetime import datetime
import sys
import os

from voice_assistant.speech.stt.base_stt import BaseSTT
from voice_assistant.speech.tts.edge_tts_generator import generate_speech
from voice_assistant.nlp.language_model import language_model
from voice_assistant.config.settings import ASSISTANT_SETTINGS

class VoiceAssistant:
    """Multilingual voice assistant with speech recognition and synthesis."""
    
    def __init__(self, stt_class=BaseSTT, silence_threshold=ASSISTANT_SETTINGS["silence_threshold_seconds"]):
        """
        Initialize the voice assistant.
        
        Args:
            stt_class: Speech-to-text class to use.
            silence_threshold: Threshold in seconds for silence detection.
        """
        # Initialize STT
        self.stt = stt_class()
        self.text_queue = queue.Queue()
        self.stop_threads = False
        
        # Initialize language model
        self.language_model = language_model
        
        # Assistant state
        self.current_mode = "general"  # general, document, or image
        self.current_file_path = None
        
        # Silence detection parameters
        self.last_speech_time = time.time()
        self.silence_threshold = silence_threshold
        self.is_listening = True
        self.is_speaking = False
        
        # Add a warm-up period to prevent immediate pausing
        self.start_time = time.time()
        self.warm_up_period = 30  # Reduced to 30 seconds warm-up period
        self.silence_detection_enabled = False  # Disable silence detection initially
        
        # Language detection
        self.current_language = "en"  # Default language
        
        # Real-time display state
        self.last_full_result = ""
        self.showing_partial = False
        
        # Conversation context tracking
        self.conversation_history = []
        self.max_history_items = 10  # Keep last 10 exchanges
        self.context_window_minutes = 15  # Consider context from last 15 minutes
    
    def silence_monitor(self):
        """Monitor for silence to pause/resume listening."""
        # Reduce silence threshold to 3 seconds for more responsive detection
        silence_threshold = 3.0  # 3 seconds
        
        while not self.stop_threads:
            # Check if we're still in the warm-up period
            elapsed_time = time.time() - self.start_time
            in_warm_up = elapsed_time < self.warm_up_period
            
            # Only enable silence detection after warm-up period
            if in_warm_up and not self.silence_detection_enabled:
                # During warm-up, print status occasionally
                if int(elapsed_time) % 10 == 0 and int(elapsed_time) > 0:
                    remaining = self.warm_up_period - elapsed_time
                    if int(remaining) % 10 == 0 and int(remaining) > 0:  # Only print every 10 seconds
                        print(f"\n[Warm-up period: {int(remaining)} seconds remaining. Silence detection disabled.]")
            elif in_warm_up and elapsed_time >= self.warm_up_period - 1:
                # About to exit warm-up
                self.silence_detection_enabled = True
                print("\n[Warm-up period complete. Silence detection enabled.]")
            
            # Only perform silence detection if enabled
            if self.silence_detection_enabled:
                if self.is_listening:
                    # Get speech state from STT
                    speech_state = self.stt.get_speech_state()
                    
                    # Check if we've been silent for too long (using 3-second threshold)
                    if (speech_state['silence_duration'] > silence_threshold and 
                        speech_state['is_silent'] and 
                        not self.is_speaking):
                        print(f"\n[Pausing listening due to silence... Silence ratio: {speech_state['silence_ratio']:.2f}]")
                        self.is_listening = False
                else:
                    # Get speech state from STT
                    speech_state = self.stt.get_speech_state()
                    
                    # Check if there's been any speech to resume
                    if speech_state['silence_ratio'] < 0.7:  # If there's significant speech
                        print(f"\n[Resuming listening... Silence ratio: {speech_state['silence_ratio']:.2f}]")
                        self.is_listening = True
            
            # Sleep for a shorter time to be more responsive
            time.sleep(0.2)
    
    def stt_processor(self):
        """Process speech-to-text results."""
        while not self.stop_threads:
            try:
                if not self.stt.result_queue.empty() and self.is_listening:
                    result = self.stt.result_queue.get()
                    if result and result.get('text', '').strip():
                        # Clear any partial results first
                        if self.showing_partial:
                            sys.stdout.write("\r" + " " * 100 + "\r")  # Clear the line
                            sys.stdout.flush()
                            self.showing_partial = False
                        
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        text = result['text']
                        lang = result['lang']
                        
                        print(f"\n[{timestamp}] [{lang}] You: {text}")
                        self.last_full_result = text
                        
                        # Update current language from the detected language
                        self.current_language = lang.split('-')[0]  # Extract primary language code
                        
                        # Add to text queue for processing
                        self.text_queue.put(text)
                        
                        # Update last speech time
                        self.last_speech_time = time.time()
                        
                        # Enable silence detection when user speaks
                        if not self.silence_detection_enabled:
                            self.silence_detection_enabled = True
                            print("\n[User speech detected. Silence detection enabled.]")
                        
                        # Reset warm-up timer when user speaks
                        self.start_time = time.time()
            except Exception as e:
                print(f"Error in STT processor: {e}")
            
            time.sleep(0.1)
    
    def check_for_commands(self, text):
        """
        Check for special commands in the text.
        
        Args:
            text: Input text to check for commands.
            
        Returns:
            Command name if a command is detected, None otherwise.
        """
        # Command to switch to document mode
        if re.search(r'\b(open|read|analyze|process)\s+(?:the\s+)?(document|pdf|file)\b', text, re.IGNORECASE):
            return "document_mode"
        
        # Command to switch to image mode
        if re.search(r'\b(look at|analyze|process|read|see)\s+(?:the\s+)?(image|picture|photo)\b', text, re.IGNORECASE):
            return "image_mode"
        
        # Command to exit document/image mode
        if re.search(r'\b(exit|leave|quit|close)\s+(?:the\s+)?(document|pdf|file|image|picture|photo|mode)\b', text, re.IGNORECASE):
            return "exit_mode"
        
        return None
    
    async def process_text_queue(self):
        """Process text from the queue."""
        while not self.stop_threads:
            try:
                if not self.text_queue.empty():
                    text = self.text_queue.get()
                    
                    # Check for commands
                    command = self.check_for_commands(text)
                    
                    if command == "exit_mode":
                        # Exit document/image mode
                        self.current_mode = "general"
                        self.current_file_path = None
                        response = "Returning to general conversation mode."
                        print(f"Assistant: {response}")
                        await generate_speech(response)
                        
                        # Add to conversation history
                        self.add_to_conversation_history("user", text)
                        self.add_to_conversation_history("assistant", response)
                    
                    else:  # General conversation
                        self.is_speaking = True
                        
                        try:
                            # Get recent conversation context
                            context = self.get_conversation_context()
                            
                            # Get response from language model with context
                            response = self.language_model.generate_response(text, context=context)
                            
                            if response:
                                print(f"Assistant: {response}")
                                # Generate and play speech
                                await generate_speech(response)
                                
                                # Add to conversation history
                                self.add_to_conversation_history("user", text)
                                self.add_to_conversation_history("assistant", response)
                            else:
                                fallback = "I'm not sure how to respond to that."
                                print(f"Assistant: {fallback}")
                                await generate_speech(fallback)
                                
                                # Add to conversation history
                                self.add_to_conversation_history("user", text)
                                self.add_to_conversation_history("assistant", fallback)
                        except Exception as e:
                            print(f"Error generating response: {e}")
                            fallback = "I'm having trouble processing that right now."
                            print(f"Assistant: {fallback}")
                            await generate_speech(fallback)
                            
                            # Add to conversation history
                            self.add_to_conversation_history("user", text)
                            self.add_to_conversation_history("assistant", fallback)
                        
                        self.is_speaking = False
            except Exception as e:
                print(f"Error in text queue processing: {e}")
            
            await asyncio.sleep(0.1)
    
    def add_to_conversation_history(self, role, text):
        """Add an exchange to the conversation history with timestamp."""
        timestamp = datetime.now()
        self.conversation_history.append({
            "role": role,
            "text": text,
            "timestamp": timestamp
        })
        
        # Trim history if it gets too long
        if len(self.conversation_history) > self.max_history_items * 2:  # *2 because each exchange has user+assistant
            self.conversation_history = self.conversation_history[-self.max_history_items*2:]
    
    def get_conversation_context(self):
        """Get recent conversation context within the time window."""
        if not self.conversation_history:
            return []
            
        # Get current time
        now = datetime.now()
        
        # Filter history to include only recent conversations
        recent_history = []
        for item in self.conversation_history:
            # Check if within time window
            time_diff = (now - item["timestamp"]).total_seconds() / 60  # in minutes
            if time_diff <= self.context_window_minutes:
                recent_history.append(item)
        
        # Format for language model
        formatted_context = []
        for item in recent_history:
            formatted_context.append({
                "role": item["role"],
                "content": item["text"]
            })
            
        return formatted_context
    
    def partial_text_monitor(self):
        """Monitor for partial text results and display them."""
        while not self.stop_threads:
            try:
                if not self.stt.partial_result_queue.empty() and self.is_listening:
                    result = self.stt.partial_result_queue.get()
                    partial_text = result.get('text', '').strip()
                    
                    if partial_text and partial_text != self.last_full_result:
                        # Show partial text with a special indicator
                        sys.stdout.write(f"\r[Partial]: {partial_text}" + " " * 10)
                        sys.stdout.flush()
                        self.showing_partial = True
                        
                        # Enable silence detection when user speaks
                        if not self.silence_detection_enabled:
                            self.silence_detection_enabled = True
                            print("\n[User speech detected. Silence detection enabled.]")
            except Exception as e:
                # Silently ignore errors in partial text monitoring
                pass
            
            time.sleep(0.1)
    
    async def run(self):
        """Run the voice assistant."""
        try:
            print("Setting up language model...")
            self.language_model.setup(ASSISTANT_SETTINGS["system_prompt"])
            
            print("Starting speech recognition...")
            try:
                self.stt.start()
            except Exception as e:
                print(f"Error starting speech recognition: {e}")
                print("Voice assistant will continue with limited functionality.")
            
            print("\n[Silence detection is disabled during startup. It will be enabled when you speak or after the warm-up period.]")
            
            # Start the silence monitor thread
            silence_thread = threading.Thread(target=self.silence_monitor)
            silence_thread.daemon = True
            silence_thread.start()
            
            # Start the STT processor thread
            stt_thread = threading.Thread(target=self.stt_processor)
            stt_thread.daemon = True
            stt_thread.start()
            
            # Start the partial text monitor thread
            partial_thread = threading.Thread(target=self.partial_text_monitor)
            partial_thread.daemon = True
            partial_thread.start()
            
            # Start the keyboard input thread
            keyboard_thread = threading.Thread(target=self.keyboard_input_thread)
            keyboard_thread.daemon = True
            keyboard_thread.start()
            
            # Welcome message
            welcome = "Welcome to the Multilingual Voice Assistant. I can understand and respond in multiple languages. How can I help you today?"
            print("\nAssistant: " + welcome)
            
            # Add to conversation history
            self.add_to_conversation_history("assistant", welcome)
            
            # Try to generate speech for welcome message
            try:
                # Add a timeout to prevent hanging
                await asyncio.wait_for(generate_speech(welcome), timeout=5)
                print("[TTS] Welcome message played successfully")
            except asyncio.TimeoutError:
                print("Welcome message audio timed out. Continuing without audio.")
            except Exception as e:
                print(f"Error generating welcome speech: {e}")
                
                # Try alternative welcome sound
                try:
                    print("Playing alternative welcome sound...")
                    # Use system beep as fallback
                    if os.name == 'nt':  # Windows
                        import winsound
                        winsound.Beep(1000, 500)  # 1000 Hz for 500 ms
                except Exception as beep_error:
                    print(f"Could not play alternative welcome sound: {beep_error}")
            
            # Process text queue
            try:
                await self.process_text_queue()
            except KeyboardInterrupt:
                print("\nShutting down...")
            except Exception as e:
                print(f"Error processing text queue: {e}")
        except Exception as e:
            print(f"Error running voice assistant: {e}")
        finally:
            # Clean up
            self.stop_threads = True
            try:
                silence_thread.join(timeout=1)
                stt_thread.join(timeout=1)
                partial_thread.join(timeout=1)
                keyboard_thread.join(timeout=1)
            except Exception as e:
                print(f"Error during cleanup: {e}")
    
    def stop(self):
        """Stop the voice assistant."""
        self.stop_threads = True
        if hasattr(self.stt, 'stop'):
            self.stt.stop()
    
    def keyboard_input_thread(self):
        """Thread to handle keyboard input as a fallback for speech recognition."""
        # Wait a moment for other initialization to complete
        time.sleep(2)
        
        print("\n" + "=" * 60)
        print("KEYBOARD INPUT MODE ENABLED")
        print("You can type commands and press Enter if speech recognition isn't working.")
        print("=" * 60 + "\n")
        
        while not self.stop_threads:
            try:
                # Prompt for input with a clear marker
                print("\n[KEYBOARD] Type a command and press Enter: ", end='', flush=True)
                text = input()
                
                # Skip empty input
                if not text.strip():
                    continue
                    
                print(f"\n[KEYBOARD] You typed: {text}")
                
                # Add to text queue for processing
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{timestamp}] You: {text}")
                self.text_queue.put(text)
                
                # Update last speech time to prevent silence detection from pausing
                self.last_speech_time = time.time()
                
                # Enable silence detection when user types
                if not self.silence_detection_enabled:
                    self.silence_detection_enabled = True
                    print("\n[User input detected. Silence detection enabled.]")
                
                # Reset warm-up timer when user types
                self.start_time = time.time()
                
                # Wait for response to be processed
                time.sleep(1)
            except EOFError:
                # This can happen when running in certain environments
                time.sleep(1)
                continue
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                print("\n[KEYBOARD] Keyboard interrupt detected. Exiting...")
                self.stop_threads = True
                break
            except Exception as e:
                print(f"\n[KEYBOARD] Error reading input: {e}")
            
            # Sleep briefly to prevent CPU overload
            time.sleep(0.1) 