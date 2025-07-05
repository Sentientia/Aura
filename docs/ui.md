# User Interface

Aura provides a user-friendly interface for interacting with the system. This page describes the UI components and how to use them.

## Overview

The Aura UI is implemented using Gradio, which provides a web-based interface for interacting with the system. The UI supports both text and speech inputs, and can provide responses in both text and speech formats.

## Components

The UI consists of the following components:

- **Input Section**: Where users can provide input through text or speech.
- **Output Section**: Where the system's responses are displayed.
- **Conversation History**: A record of the conversation between the user and the system.
- **Model Selection**: Options to select different ASR, TTS, and LLM models.
- **Settings**: Configuration options for the system.

## Usage

To launch the UI, run the following command:

```bash
python ui/local_speech_app.py
```

This will start a local web server that you can access in your browser.

## Speech Interface

The speech interface allows users to interact with the system using natural speech. To use the speech interface:

1. Click the microphone button to start recording.
2. Speak your request or question.
3. Click the stop button to stop recording.
4. The system will transcribe your speech, process your request, and provide a response.
5. If speech output is enabled, the system will also speak the response.

## Text Interface

The text interface allows users to interact with the system using text. To use the text interface:

1. Type your request or question in the input field.
2. Press Enter or click the submit button.
3. The system will process your request and provide a response.

## Model Selection

The UI allows users to select different models for ASR, TTS, and LLM:

- **ASR Models**: Options include Whisper, OWSM, and other speech recognition models.
- **TTS Models**: Options include ESPnet and other speech synthesis models.
- **LLM Models**: Options include different language models for natural language understanding and generation.

To select a model:

1. Click the dropdown menu for the desired component (ASR, TTS, or LLM).
2. Select the desired model from the list.
3. The system will use the selected model for future interactions.

## Settings

The UI provides various settings to customize the system's behavior:

- **Speech Input**: Enable or disable speech input.
- **Speech Output**: Enable or disable speech output.
- **Conversation History**: Clear the conversation history.
- **Debug Mode**: Enable or disable debug information.

To access the settings:

1. Click the settings button in the UI.
2. Adjust the settings as desired.
3. Click the save button to apply the changes.

## Example Interaction

Here's an example of how to interact with the system through the UI:

1. User speaks or types: "What's the weather like today?"
2. System processes the request and responds: "I'll check the weather for you. Where are you located?"
3. User speaks or types: "New York City"
4. System uses the Web Search Action to find weather information for New York City.
5. System responds with the current weather conditions and forecast for New York City.

This interaction demonstrates how the system can engage in a natural conversation with the user, gather necessary information, and provide helpful responses.