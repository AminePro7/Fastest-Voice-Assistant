"""
Base Speech-to-Text (STT) implementation.

This module provides a basic STT implementation using speech_recognition.
"""

import speech_recognition as sr
import threading
import queue
import time
from datetime import datetime
import numpy as np
from langdetect import detect
import os

from voice_assistant.config.settings import STT_SETTINGS, SUPPORTED_LANGUAGES

class BaseSTT:
    """Base class for speech-to-text functionality."""
    
    def __init__(self, 
                 energy_threshold=STT_SETTINGS["energy_threshold"],
                 sample_rate=STT_SETTINGS["sample_rate"],
                 silence_threshold=STT_SETTINGS["silence_threshold"]):
        """
        Initialize the STT system.
        
        Args:
            energy_threshold: Energy threshold for speech detection.
            sample_rate: Audio sample rate in Hz.
            silence_threshold: Ratio of frames that need to be silent to consider a segment as silence.
        """
        # List available microphones to help diagnose issues
        try:
            print("\n[STT] Available microphones:")
            mic_list = sr.Microphone.list_microphone_names()
            for i, mic_name in enumerate(mic_list):
                print(f"[STT] Microphone {i}: {mic_name}")
            
            # If no microphones are found, print a warning
            if not mic_list:
                print("[STT] WARNING: No microphones found!")
        except Exception as mic_list_error:
            print(f"[STT] Error listing microphones: {mic_list_error}")
        
        # Find the best microphone to use
        self._best_mic_index = self.find_best_microphone()
        
        # Initialize speech recognition components
        self.recognizer = sr.Recognizer()
        
        # Create microphone with the best device index if available
        if self._best_mic_index is not None:
            self.microphone = sr.Microphone(device_index=self._best_mic_index, sample_rate=sample_rate)
            print(f"[STT] Using microphone with index {self._best_mic_index} and sample rate {sample_rate}")
        else:
            self.microphone = sr.Microphone(sample_rate=sample_rate)
            print(f"[STT] Using default microphone with sample rate {sample_rate}")
            
        self.result_queue = queue.Queue()
        self.partial_result_queue = queue.Queue()  # Queue for partial results
        self.stop_thread = False
        self.is_listening = True
        self.last_audio_time = time.time()
        self.last_speech_time = time.time()
        
        # Speech detection state
        self.speech_detected = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.total_frames = 0
        self.silence_ratio = 1.0  # Start with silence
        self.silence_threshold = silence_threshold
        
        # Optimize recognition settings - lower energy threshold for better sensitivity
        # Use a much lower energy threshold to make it more sensitive
        self.recognizer.energy_threshold = 300  # Use an extremely low threshold for maximum sensitivity
        self.recognizer.dynamic_energy_threshold = False  # Disable dynamic adjustment for consistent sensitivity
        self.recognizer.pause_threshold = 0.3  # Very short pause threshold for better responsiveness
        self.recognizer.phrase_threshold = 0.1  # Lower phrase threshold for better sensitivity
        self.recognizer.non_speaking_duration = 0.2  # Shorter non-speaking duration for better responsiveness
        
        # Store minimum threshold to prevent dropping too low
        self.min_energy_threshold = 300  # Use an extremely low minimum threshold for maximum sensitivity
        
        # Language detection
        self.supported_languages = list(SUPPORTED_LANGUAGES.values())
        self.last_detected_language = 'en-US'  # Default language
        
        # Real-time recognition
        self.current_partial_text = ""
        
        # Debug mode
        self.debug_mode = True
        
        # Active listening control
        self.active_listening_running = False
        
        print(f"[STT] Initialized with energy threshold: {self.recognizer.energy_threshold}")
        print(f"[STT] Dynamic energy threshold: {self.recognizer.dynamic_energy_threshold}")
        print(f"[STT] Minimum energy threshold: {self.min_energy_threshold}")
        print(f"[STT] Pause threshold: {self.recognizer.pause_threshold}")
        print(f"[STT] Phrase threshold: {self.recognizer.phrase_threshold}")
        print(f"[STT] Non-speaking duration: {self.recognizer.non_speaking_duration}")
    
    def adjust_for_ambient_noise(self, duration=STT_SETTINGS["ambient_noise_duration"]):
        """Adjust the recognizer for ambient noise."""
        print("Adjusting for ambient noise... Please wait.")
        
        # Find the best microphone to use if not already set
        best_mic_index = getattr(self, '_best_mic_index', None)
        if best_mic_index is None:
            best_mic_index = self.find_best_microphone()
            self._best_mic_index = best_mic_index
        
        # Skip the actual ambient noise adjustment and use a fixed low threshold
        # This is more reliable than trying to detect ambient noise
        self.recognizer.energy_threshold = 300  # Use an extremely low fixed threshold for maximum sensitivity
        print(f"Using fixed low energy threshold: {self.recognizer.energy_threshold}")
        
        # Still try to do the actual adjustment in a separate thread
        # Create a flag for the timeout
        adjustment_complete = False
        
        # Define the adjustment function
        def perform_adjustment():
            nonlocal adjustment_complete
            try:
                # Use the best microphone if available
                if best_mic_index is not None:
                    mic = sr.Microphone(device_index=best_mic_index)
                    print(f"Using microphone with index {best_mic_index} for ambient noise adjustment")
                else:
                    mic = self.microphone
                    
                with mic as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=duration)
                    # Ensure threshold doesn't go below minimum but keep it sensitive
                    self.recognizer.energy_threshold = max(self.recognizer.energy_threshold, self.min_energy_threshold)
                    # But also make sure it's not too high - force a very low threshold
                    self.recognizer.energy_threshold = min(self.recognizer.energy_threshold, 300)
                adjustment_complete = True
            except Exception as e:
                print(f"Error during ambient noise adjustment: {e}")
                # Keep using the fixed threshold
                adjustment_complete = True
        
        # Start the adjustment in a separate thread
        adjustment_thread = threading.Thread(target=perform_adjustment)
        adjustment_thread.daemon = True
        adjustment_thread.start()
        
        # Wait for the adjustment with a timeout
        timeout = 5  # 5 seconds timeout (reduced from 10)
        start_time = time.time()
        while not adjustment_complete and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        # Check if adjustment completed or timed out
        if adjustment_complete:
            print(f"Ambient noise adjustment complete! Energy threshold set to: {self.recognizer.energy_threshold}")
        else:
            print("Ambient noise adjustment timed out. Using default threshold.")
            self.recognizer.energy_threshold = self.min_energy_threshold
            print(f"Set energy threshold to default: {self.recognizer.energy_threshold}")
            
        # Force a very low threshold regardless of what happened
        self.recognizer.energy_threshold = 300
        print(f"Forced energy threshold to: {self.recognizer.energy_threshold} for maximum sensitivity")
    
    def process_audio(self, audio_data):
        """
        Process audio data to detect speech.
        
        Args:
            audio_data: Audio data from the microphone.
            
        Returns:
            True if speech was detected, False otherwise.
        """
        try:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(audio_data.frame_data, dtype=np.int16)
            
            # Calculate RMS energy
            energy = np.sqrt(np.mean(np.square(audio_array, dtype=np.float32)))
            
            # Update speech/silence counters
            self.total_frames += 1
            
            if energy > self.recognizer.energy_threshold * 0.8:  # More sensitive threshold
                self.speech_frames += 1
                self.last_speech_time = time.time()
                is_speech_detected = True
                if self.debug_mode:
                    print(f"\n[DEBUG] Speech detected! Energy: {energy:.0f}, Threshold: {self.recognizer.energy_threshold:.0f}")
            else:
                self.silence_frames += 1
                is_speech_detected = False
            
            # Update silence ratio
            if self.total_frames > 0:
                self.silence_ratio = self.silence_frames / self.total_frames
            
            # Reset counters periodically to adapt to changing conditions
            if self.total_frames > 100:  # Reset after 100 frames
                self.total_frames = 0
                self.silence_frames = 0
                self.speech_frames = 0
            
            return is_speech_detected
            
        except Exception as e:
            print(f"Error in audio processing: {e}")
            return False
    
    def audio_callback(self, recognizer, audio_data):
        """Callback for when audio is detected"""
        try:
            # Always process the audio regardless of speech detection
            speech_detected = self.process_audio(audio_data)
            self.audio_queue.put(audio_data)
            
            if speech_detected:
                print(f"\n[SYSTEM]: Speech detected (Silence ratio: {self.silence_ratio:.2f})")
                self.last_audio_time = time.time()
                
                # Try to get real-time partial recognition
                try:
                    # Use Google's recognizer with partial results
                    partial_text = self.recognizer.recognize_google(
                        audio_data, 
                        language=self.last_detected_language,
                        show_all=True
                    )
                    
                    # Check if we got a valid result with alternatives
                    if isinstance(partial_text, dict) and 'alternative' in partial_text:
                        # Get the most likely text
                        if partial_text['alternative']:
                            best_guess = partial_text['alternative'][0]['transcript']
                            if best_guess:
                                self.current_partial_text = best_guess
                                self.partial_result_queue.put({
                                    'text': best_guess,
                                    'time': datetime.now().strftime("%H:%M:%S.%f")[:-4]
                                })
                except Exception as e:
                    if self.debug_mode:
                        print(f"\n[DEBUG] Partial recognition failed: {e}")
                    # Partial recognition failed, but that's okay
                    pass
        except Exception as e:
            print(f"Error in audio callback: {e}")
    
    def detect_language_from_text(self, text):
        """
        Detect language from text using langdetect.
        
        Args:
            text: Text to detect language from.
            
        Returns:
            Language code in Google Speech Recognition format.
        """
        try:
            detected_lang_code = detect(text)
            lang_map = SUPPORTED_LANGUAGES
            return lang_map.get(detected_lang_code, self.last_detected_language)
        except:
            # If detection fails, return the last known language
            return self.last_detected_language
    
    def recognize_worker(self):
        """Worker thread for processing audio chunks"""
        while not self.stop_thread:
            if not self.audio_queue.empty():
                audio = self.audio_queue.get()
                try:
                    # First try to recognize with the last detected language
                    try:
                        text = self.recognizer.recognize_google(audio, language=self.last_detected_language)
                        if text.strip():
                            # Verify language with langdetect
                            detected_lang = self.detect_language_from_text(text)
                            
                            # If detected language is different, try again with the detected language
                            if detected_lang != self.last_detected_language:
                                try:
                                    text = self.recognizer.recognize_google(audio, language=detected_lang)
                                    self.last_detected_language = detected_lang
                                except:
                                    # If recognition fails with detected language, keep the original text
                                    pass
                            
                            current_time = datetime.now().strftime("%H:%M:%S.%f")[:-4]
                            self.result_queue.put({
                                'text': text,
                                'lang': self.last_detected_language,
                                'time': current_time,
                                'energy': self.recognizer.energy_threshold
                            })
                            # Update last audio time when we successfully recognize speech
                            self.last_audio_time = time.time()
                            
                            if self.debug_mode:
                                print(f"\n[DEBUG] Recognition successful: '{text}' in {self.last_detected_language}")
                    except sr.UnknownValueError:
                        if self.debug_mode:
                            print(f"\n[DEBUG] Recognition failed with {self.last_detected_language}, trying other languages")
                        
                        # If recognition fails with the last language, try with other languages
                        for lang in self.supported_languages:
                            if lang == self.last_detected_language:
                                continue  # Skip the language we already tried
                            
                            try:
                                text = self.recognizer.recognize_google(audio, language=lang)
                                if text.strip():
                                    # Verify language with langdetect
                                    detected_lang = self.detect_language_from_text(text)
                                    self.last_detected_language = detected_lang
                                    
                                    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-4]
                                    self.result_queue.put({
                                        'text': text,
                                        'lang': detected_lang,
                                        'time': current_time,
                                        'energy': self.recognizer.energy_threshold
                                    })
                                    # Update last audio time when we successfully recognize speech
                                    self.last_audio_time = time.time()
                                    
                                    if self.debug_mode:
                                        print(f"\n[DEBUG] Recognition successful with alternate language: '{text}' in {detected_lang}")
                                    
                                    break
                            except sr.UnknownValueError:
                                continue
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
            time.sleep(0.05)
    
    def partial_result_processor(self):
        """Process partial recognition results to show real-time feedback."""
        last_partial = ""
        while not self.stop_thread:
            try:
                if not self.partial_result_queue.empty():
                    result = self.partial_result_queue.get()
                    partial_text = result['text']
                    
                    # Only update if the text has changed
                    if partial_text != last_partial:
                        # Print the partial result with a special indicator
                        print(f"\r[Partial]: {partial_text}", end="", flush=True)
                        last_partial = partial_text
            except Exception as e:
                pass  # Silently ignore errors in partial processing
                
            time.sleep(0.1)
    
    def active_listening_loop(self):
        """Continuously listen for speech in a separate thread."""
        self.active_listening_running = True
        print("\n[ACTIVE LISTENING] Starting active listening loop...")
        
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        # Try to find the best microphone to use
        best_mic_index = self.find_best_microphone()
        if best_mic_index is not None:
            print(f"[ACTIVE LISTENING] Using microphone with index: {best_mic_index}")
        else:
            print("[ACTIVE LISTENING] No specific microphone selected, will use default")
        
        # Create a single recognizer and microphone instance to reuse
        try:
            recognizer = sr.Recognizer()
            # Use an extremely low threshold for maximum sensitivity
            recognizer.energy_threshold = 300  # Even lower threshold for better sensitivity
            recognizer.dynamic_energy_threshold = False  # Disable dynamic adjustment for active listening
            recognizer.pause_threshold = 0.3  # Very short pause threshold
            
            # Create a single microphone instance with the best mic index
            if best_mic_index is not None:
                microphone = sr.Microphone(device_index=best_mic_index)
            else:
                microphone = sr.Microphone()
            
            print(f"[ACTIVE LISTENING] Created microphone with device index: {microphone.device_index}")
            print(f"[ACTIVE LISTENING] Using energy threshold: {recognizer.energy_threshold}")
            
            # Test the microphone to make sure it's working
            try:
                print("[ACTIVE LISTENING] Testing microphone...")
                with microphone as source:
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=1)
                print("[ACTIVE LISTENING] Microphone test successful!")
            except Exception as test_error:
                print(f"[ACTIVE LISTENING] Microphone test failed: {test_error}")
                # Continue anyway, it might still work
        except Exception as init_error:
            print(f"[ACTIVE LISTENING] Error initializing: {init_error}")
            print("[ACTIVE LISTENING] Will try to initialize in the loop")
            recognizer = None
            microphone = None
        
        while not self.stop_thread and self.active_listening_running:
            try:
                # If we don't have a recognizer or microphone, try to create them
                if not recognizer or not microphone:
                    try:
                        recognizer = sr.Recognizer()
                        recognizer.energy_threshold = 300  # Even lower threshold
                        recognizer.dynamic_energy_threshold = False
                        recognizer.pause_threshold = 0.3
                        
                        # Try to use the best microphone if available
                        if best_mic_index is not None:
                            microphone = sr.Microphone(device_index=best_mic_index)
                        else:
                            microphone = sr.Microphone()
                            
                        print(f"[ACTIVE LISTENING] Re-created microphone with device index: {microphone.device_index}")
                    except Exception as reinit_error:
                        print(f"[ACTIVE LISTENING] Error re-initializing: {reinit_error}")
                        time.sleep(2)
                        continue
                
                # Use a try-except block for microphone access
                try:
                    print("\n[ACTIVE LISTENING] Listening for speech...")
                    print(f"[ACTIVE LISTENING] Energy threshold: {recognizer.energy_threshold}")
                    
                    # Use a with block for the microphone
                    with microphone as source:
                        # Adjust for ambient noise briefly before each listen
                        try:
                            print("[ACTIVE LISTENING] Quick adjustment for ambient noise...")
                            recognizer.adjust_for_ambient_noise(source, duration=0.5)
                            print(f"[ACTIVE LISTENING] Adjusted energy threshold: {recognizer.energy_threshold}")
                            
                            # Force a very low threshold regardless of ambient noise
                            if recognizer.energy_threshold > 300:
                                recognizer.energy_threshold = 300
                                print(f"[ACTIVE LISTENING] Forced energy threshold to: {recognizer.energy_threshold}")
                        except Exception as adjust_error:
                            print(f"[ACTIVE LISTENING] Error adjusting for ambient noise: {adjust_error}")
                            # Force a very low threshold
                            recognizer.energy_threshold = 300
                            print(f"[ACTIVE LISTENING] Forced energy threshold to: {recognizer.energy_threshold}")
                        
                        # Use a timeout for listen to prevent hanging
                        try:
                            print("[ACTIVE LISTENING] Waiting for speech...")
                            # Increased timeout for better chance of capturing speech
                            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                            print("\n[ACTIVE LISTENING] Speech detected, processing...")
                            
                            # Reset consecutive errors counter on success
                            consecutive_errors = 0
                            
                            # Try to recognize the speech
                            try:
                                # Try multiple recognition services for better results
                                text = None
                                
                                # First try Google
                                try:
                                    print("[ACTIVE LISTENING] Trying Google recognition...")
                                    text = recognizer.recognize_google(audio, language=self.last_detected_language)
                                except Exception as google_error:
                                    print(f"[ACTIVE LISTENING] Google recognition error: {google_error}")
                                    
                                # If Google fails, try Sphinx (offline)
                                if not text:
                                    try:
                                        print("[ACTIVE LISTENING] Trying Sphinx recognition...")
                                        text = recognizer.recognize_sphinx(audio)
                                        print("[ACTIVE LISTENING] Used Sphinx (offline) recognition")
                                    except Exception as sphinx_error:
                                        print(f"[ACTIVE LISTENING] Sphinx recognition error: {sphinx_error}")
                                
                                if text:
                                    print(f"\n[ACTIVE LISTENING] Recognized: '{text}'")
                                    
                                    # Put the result in the queue
                                    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-4]
                                    self.result_queue.put({
                                        'text': text,
                                        'lang': self.last_detected_language,
                                        'time': current_time,
                                        'energy': recognizer.energy_threshold
                                    })
                                else:
                                    print("\n[ACTIVE LISTENING] Could not recognize speech")
                            except sr.UnknownValueError:
                                print("\n[ACTIVE LISTENING] Could not understand audio")
                            except Exception as e:
                                print(f"\n[ACTIVE LISTENING] Recognition error: {e}")
                        except sr.WaitTimeoutError:
                            print("\n[ACTIVE LISTENING] No speech detected, continuing...")
                        except Exception as listen_error:
                            print(f"\n[ACTIVE LISTENING] Listening error: {listen_error}")
                            consecutive_errors += 1
                except Exception as source_error:
                    print(f"\n[ACTIVE LISTENING] Microphone source error: {source_error}")
                    consecutive_errors += 1
                    # Recreate the microphone on next iteration
                    microphone = None
                    recognizer = None
                    time.sleep(2)
            except Exception as e:
                print(f"\n[ACTIVE LISTENING] Error in active listening loop: {e}")
                consecutive_errors += 1
                # Add a short sleep to prevent CPU overload in case of repeated errors
                time.sleep(1)
            
            # Check if we need to reset the recognizer due to too many errors
            if consecutive_errors >= max_consecutive_errors:
                print("\n[ACTIVE LISTENING] Too many consecutive errors. Resetting recognizer...")
                # Reset the recognizer and microphone
                try:
                    recognizer = sr.Recognizer()
                    recognizer.energy_threshold = 300  # Even lower threshold
                    recognizer.dynamic_energy_threshold = False
                    recognizer.pause_threshold = 0.3
                    
                    # Try to use the best microphone if available
                    if best_mic_index is not None:
                        microphone = sr.Microphone(device_index=best_mic_index)
                    else:
                        microphone = sr.Microphone()
                        
                    print(f"[ACTIVE LISTENING] Reset microphone with device index: {microphone.device_index}")
                except Exception as reset_error:
                    print(f"[ACTIVE LISTENING] Error resetting: {reset_error}")
                    microphone = None
                    recognizer = None
                
                consecutive_errors = 0
                time.sleep(3)  # Sleep longer to allow system to recover
            
            # Sleep briefly before the next iteration
            time.sleep(0.5)
    
    def find_best_microphone(self):
        """Find the best microphone to use based on the available devices."""
        try:
            # Get list of microphone names
            mic_names = sr.Microphone.list_microphone_names()
            
            # Try to use microphone index 7 or 14 (EPOS IMPACT 660 ANC) which seems to be a good headset
            for specific_index in [7, 14, 24, 25, 28]:
                if specific_index < len(mic_names):
                    print(f"[STT] Trying specific microphone {specific_index}: {mic_names[specific_index]}")
                    return specific_index
            
            # Look for specific keywords in microphone names
            # Prioritize headset microphones and actual microphones over other devices
            priority_keywords = [
                "headset mic", "headset microphone", 
                "mic input", "microphone input",
                "microphone (", "mic (",
                "input"
            ]
            
            # First pass: look for priority keywords
            for keyword in priority_keywords:
                for i, name in enumerate(mic_names):
                    if keyword.lower() in name.lower():
                        print(f"[STT] Selected microphone {i}: {name} (matched keyword: {keyword})")
                        return i
            
            # Second pass: look for any microphone that's not an output device
            avoid_keywords = ["output", "speaker", "playback", "sound mapper - output"]
            for i, name in enumerate(mic_names):
                if not any(keyword.lower() in name.lower() for keyword in avoid_keywords):
                    print(f"[STT] Selected microphone {i}: {name} (not an output device)")
                    return i
            
            # If we have microphones but couldn't find a good one, use the first one
            if mic_names:
                print(f"[STT] Selected first available microphone 0: {mic_names[0]}")
                return 0
                
            # If no microphones found, return None
            print("[STT] No suitable microphone found")
            return None
        except Exception as e:
            print(f"[STT] Error finding best microphone: {e}")
            return None
    
    def start(self):
        """Start the STT system"""
        try:
            # Find the best microphone to use
            best_mic_index = self.find_best_microphone()
            
            # Initialize audio queue
            self.audio_queue = queue.Queue()
            
            # Adjust for ambient noise
            self.adjust_for_ambient_noise()
            
            # Start multiple recognition workers for parallel processing
            for _ in range(3):  # Create 3 worker threads
                threading.Thread(target=self.recognize_worker, daemon=True).start()
            
            # Start the partial result processor
            threading.Thread(target=self.partial_result_processor, daemon=True).start()
            
            # Start the active listening loop in a separate thread
            active_thread = threading.Thread(target=self.active_listening_loop, daemon=True)
            active_thread.start()

            # Try to start the background listener, but don't worry if it fails
            # The active listening loop will still work
            try:
                print("Starting background listener...")
                
                # Create a new microphone instance specifically for background listening
                # This is important to avoid conflicts with the active listening loop
                try:
                    if best_mic_index is not None:
                        background_mic = sr.Microphone(device_index=best_mic_index)
                    else:
                        background_mic = sr.Microphone()
                        
                    print(f"Created background microphone with device index: {background_mic.device_index}")
                    
                    # Start listening in the background
                    self.stop_listening = self.recognizer.listen_in_background(
                        background_mic,
                        self.audio_callback,
                        phrase_time_limit=10
                    )
                    print("Background listener started successfully.")
                except Exception as mic_error:
                    print(f"Error creating background microphone: {mic_error}")
                    print("Falling back to active listening only.")
            except Exception as e:
                print(f"Error starting background listener: {e}")
                print("Falling back to active listening only.")
            
            print("Started listening with real-time word recognition...")
            print(f"Current energy threshold: {self.recognizer.energy_threshold}")
            print("Speak something!")
        except Exception as e:
            print(f"Error starting STT system: {e}")
            print("Voice recognition may not work properly. Using fallback mode.")
            
            # Set a default energy threshold
            self.recognizer.energy_threshold = self.min_energy_threshold
            
            # Start active listening as a fallback
            try:
                threading.Thread(target=self.active_listening_loop, daemon=True).start()
                print("Started in fallback mode with active listening only.")
            except Exception as e2:
                print(f"Critical error: {e2}")
                print("Voice recognition is not available.")
    
    def get_silence_duration(self):
        """Get the duration of silence in seconds since last speech"""
        return time.time() - self.last_speech_time
    
    def get_speech_state(self):
        """Get the current speech state"""
        return {
            'silence_ratio': self.silence_ratio,
            'is_silent': self.silence_ratio > self.silence_threshold,
            'last_speech_time': self.last_speech_time,
            'silence_duration': self.get_silence_duration()
        }
    
    def stop(self):
        """Stop the STT system"""
        self.stop_thread = True
        self.active_listening_running = False
        if hasattr(self, 'stop_listening'):
            self.stop_listening(wait_for_stop=False) 