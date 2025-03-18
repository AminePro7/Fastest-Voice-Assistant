# Multilingual Voice Assistant

A powerful voice assistant that supports multiple languages for speech recognition, language model responses, and text-to-speech output. The assistant automatically detects the language of the user's query and responds in the same language.

## Features

- **Multilingual Support**: Automatically detects and responds in the user's language
- **Real-time Speech Recognition**: Uses `speech_recognition` for accurate voice input
- **Enhanced Voice Activity Detection**: Robust speech detection in noisy environments
- **Advanced Language Model**: Powered by Gemini 1.5 Pro for intelligent responses
- **Natural Text-to-Speech**: Generates human-like speech in multiple languages

## Supported Languages

The voice assistant supports the following languages for both input and output:

- English (en)
- Spanish (es)
- French (fr)
- Arabic (ar)
- German (de)
- Italian (it)
- Japanese (ja)
- Chinese (zh)

## Project Structure

```
voice_assistant/
├── core/                  # Core assistant functionality
│   ├── assistant.py       # Main voice assistant class
│   └── ...
├── speech/                # Speech processing components
│   ├── stt/               # Speech-to-Text components
│   │   ├── base_stt.py    # Base STT implementation
│   │   └── ...
│   ├── tts/               # Text-to-Speech components
│   │   ├── edge_tts_generator.py  # TTS implementation
│   │   └── ...
│   └── ...
├── nlp/                   # Natural Language Processing
│   ├── language_model.py  # Language model implementation
│   ├── rag/               # Retrieval-Augmented Generation
│   │   └── ...
│   └── ...
├── utils/                 # Utility functions
│   └── ...
├── config/                # Configuration
│   ├── settings.py        # Settings and parameters
│   └── ...
├── scripts/               # Executable scripts
│   ├── run.py             # Main script to run the assistant
│   └── ...
├── data/                  # Data files
│   ├── models/            # Model files
│   ├── documents/         # Document files for RAG
│   └── ...
└── docs/                  # Documentation
    └── ...
```

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/voice_assistant.git
cd voice_assistant
```

2. Install the package:
```
pip install -e .
```

## Usage

### Running the Voice Assistant

```python
from voice_assistant.core.assistant import VoiceAssistant

async def main():
    assistant = VoiceAssistant()
    await assistant.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

Or use the command-line script:

```
python -m voice_assistant.scripts.run
```

## Development

### Adding a New STT Implementation

1. Create a new file in `voice_assistant/speech/stt/`
2. Subclass `BaseSTT` from `voice_assistant.speech.stt.base_stt`
3. Override the necessary methods

### Adding a New TTS Implementation

1. Create a new file in `voice_assistant/speech/tts/`
2. Create a class with a `generate_and_play_speech` method

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini for the language model
- Microsoft for edge-tts
- The developers of speech_recognition, pygame, and other libraries used in this project 