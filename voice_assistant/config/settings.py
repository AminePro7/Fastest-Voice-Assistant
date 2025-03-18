"""
Configuration settings for the voice assistant.
"""

# Speech recognition settings
STT_SETTINGS = {
    "energy_threshold": 4000,
    "sample_rate": 16000,
    "silence_threshold": 0.9,
    "pause_threshold": 0.8,
    "phrase_threshold": 0.3,
    "non_speaking_duration": 0.4,
    "ambient_noise_duration": 3,
    "min_energy_threshold": 2000,
}

# Text-to-speech settings
TTS_SETTINGS = {
    # Language to voice mapping for TTS
    "voice_mapping": {
        'en': 'en-US-ChristopherNeural',
        'es': 'es-ES-AlvaroNeural',
        'fr': 'fr-FR-HenriNeural',
        'ar': 'ar-SA-HamedNeural',
        'de': 'de-DE-ConradNeural',
        'it': 'it-IT-DiegoNeural',
        'ja': 'ja-JP-KeitaNeural',
        'zh': 'zh-CN-YunxiNeural',
    }
}

# Language model settings
LLM_SETTINGS = {
    "model_name": "gemini-1.5-pro",
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
    "safety_settings": [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
}

# Supported languages
SUPPORTED_LANGUAGES = {
    'en': 'en-US',
    'fr': 'fr-FR',
    'es': 'es-ES',
    'de': 'de-DE',
    'it': 'it-IT',
    'ar': 'ar-AR',
    'zh': 'zh-CN',
    'ja': 'ja-JP'
}

# Paths
PATHS = {
    "api_key_path": "keys/.gemini_api_key.txt",
    "models_dir": "data/models",
    "documents_dir": "data/documents",
    "temp_dir": None  # Will be set to system temp directory at runtime
}

# Assistant settings
ASSISTANT_SETTINGS = {
    "silence_threshold_seconds": 5.0,
    "system_prompt": "You are a helpful multilingual voice assistant. Keep your responses concise and conversational. Always respond in the same language as the user's query."
} 