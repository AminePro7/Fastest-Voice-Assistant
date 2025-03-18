"""
Language model implementation using Google's Gemini API.

This module provides language model functionality for generating responses.
"""

import os
import google.generativeai as genai

from voice_assistant.config.settings import LLM_SETTINGS, PATHS

class LanguageModel:
    """Language model using Google's Gemini API."""
    
    def __init__(self, 
                 model_name=LLM_SETTINGS["model_name"],
                 temperature=LLM_SETTINGS["temperature"],
                 top_p=LLM_SETTINGS["top_p"],
                 top_k=LLM_SETTINGS["top_k"],
                 max_output_tokens=LLM_SETTINGS["max_output_tokens"],
                 safety_settings=LLM_SETTINGS["safety_settings"]):
        """
        Initialize the language model.
        
        Args:
            model_name: Name of the Gemini model to use.
            temperature: Temperature for response generation.
            top_p: Top-p value for response generation.
            top_k: Top-k value for response generation.
            max_output_tokens: Maximum number of tokens to generate.
            safety_settings: Safety settings for the model.
        """
        self.api_key = None
        self.model = None
        self.chat_session = None
        
        # Model settings
        self.model_name = model_name
        self.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
        }
        self.safety_settings = safety_settings
    
    def get_api_key(self):
        """Get API key from file or environment variable."""
        # Try to get from file first
        key_path = PATHS["api_key_path"]
        if os.path.exists(key_path):
            with open(key_path, "r") as f:
                return f.read().strip()
        
        # Try environment variable
        return os.environ.get("GEMINI_API_KEY")
    
    def setup(self, system_prompt=None):
        """
        Set up the language model.
        
        Args:
            system_prompt: System prompt to initialize the chat session.
        """
        self.api_key = self.get_api_key()
        if not self.api_key:
            self.api_key = input("Enter your Gemini API Key: ")
            save = input("Would you like to save this API key for future use? (y/n): ")
            if save.lower() == 'y':
                os.makedirs(os.path.dirname(PATHS["api_key_path"]), exist_ok=True)
                with open(PATHS["api_key_path"], "w") as f:
                    f.write(self.api_key)
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Set up the model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        
        # Start a chat session
        if system_prompt:
            self.chat_session = self.model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [system_prompt]
                    },
                    {
                        "role": "model",
                        "parts": ["I understand and I'm ready to assist as instructed."]
                    }
                ]
            )
        else:
            self.chat_session = self.model.start_chat()
        
        print(f"Gemini model '{self.model_name}' initialized successfully!")
        return self.chat_session
    
    def generate_response(self, text, context=None):
        """
        Generate a response to the given text.
        
        Args:
            text: Input text to generate a response for.
            context: Optional conversation context as a list of message objects.
            
        Returns:
            Generated response text.
        """
        if not self.chat_session:
            raise ValueError("Language model not initialized. Call setup() first.")
        
        try:
            # If we have context, use it to create a new chat session with history
            if context and len(context) > 0:
                try:
                    # Create a new chat session with the same parameters
                    history_session = genai.GenerativeModel(
                        model_name=self.model_name,
                        generation_config=self.generation_config,
                        safety_settings=self.safety_settings
                    ).start_chat(history=[])
                    
                    # Add system prompt if it exists
                    if hasattr(self, 'system_prompt') and self.system_prompt:
                        history_session.history.append({"role": "system", "parts": [self.system_prompt]})
                    
                    # Add conversation history
                    for message in context:
                        role = message["role"]
                        content = message["content"]
                        
                        # Convert 'user' and 'assistant' roles to 'user' and 'model'
                        if role == "assistant":
                            role = "model"
                        
                        history_session.history.append({"role": role, "parts": [content]})
                    
                    # Send the current message
                    response = history_session.send_message(text)
                    return response.text
                except Exception as context_error:
                    print(f"Error using context: {context_error}")
                    print("Falling back to contextless response")
                    # Fall back to regular response without context
                    response = self.chat_session.send_message(text)
                    return response.text
            else:
                # Use the regular chat session without context
                response = self.chat_session.send_message(text)
                return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

# Singleton instance for easy access
language_model = LanguageModel()

# Convenience function for generating responses
def generate_response(text, system_prompt=None, context=None):
    """
    Generate a response to the given text.
    
    Args:
        text: Input text to generate a response for.
        system_prompt: Optional system prompt to use.
        context: Optional conversation context.
        
    Returns:
        Generated response text.
    """
    if not language_model.chat_session:
        language_model.setup(system_prompt)
    return language_model.generate_response(text, context=context) 