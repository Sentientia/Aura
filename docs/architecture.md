# Architecture

Aura is built with a modular architecture that allows for easy extension and customization. This page provides an overview of the system architecture and how the different components interact with each other.

## System Overview

![Aura System Architecture](images/aura_system_white.png)

The Aura system consists of several key components:

1. **User Interface**: The entry point for user interactions, which can be text or speech-based.
2. **Speech Processing**: Handles speech recognition (ASR) and speech synthesis (TTS).
3. **Agent Controller**: Manages the state of the conversation and coordinates between different components.
4. **Agent Hub**: Contains different agent implementations for different use cases.
5. **Action Framework**: Provides a set of actions that the agent can take to fulfill user requests.
6. **Dialog State Tracking**: Keeps track of the conversation context to enable more natural interactions.
7. **Language Model Integration**: Connects to external language models for natural language understanding and generation.

## Component Details

### User Interface

The user interface is implemented using Gradio, which provides a web-based interface for interacting with the system. The UI supports both text and speech inputs, and can provide responses in both text and speech formats.

### Speech Processing

The speech processing component includes:

- **Automatic Speech Recognition (ASR)**: Converts speech to text using models like Whisper or OWSM.
- **Text-to-Speech (TTS)**: Converts text to speech using models like ESPnet.

### Agent Controller

The agent controller is responsible for:

- Managing the state of the conversation
- Coordinating between different components
- Handling the flow of information between the user and the agent

### Agent Hub

The agent hub contains different agent implementations:

- **Chat Agent**: A general-purpose agent for conversational interactions.
- **QA Agent**: A specialized agent for question-answering tasks.

### Action Framework

The action framework provides a set of actions that the agent can take:

- **Chat Action**: Engage in conversation with the user.
- **Web Search Action**: Search the web for information.
- **Calendar Action**: Manage calendar events.
- **Email Action**: Send and manage emails.
- **Contact Action**: Manage contact information.

### Dialog State Tracking

The dialog state tracking component keeps track of the conversation context, including:

- User preferences
- Previous interactions
- Current conversation state

### Language Model Integration

The language model integration component connects to external language models for:

- Natural language understanding
- Natural language generation
- Reasoning about user requests

## Data Flow

1. The user interacts with the system through the UI, providing either text or speech input.
2. If the input is speech, it is converted to text by the ASR component.
3. The text is passed to the agent controller, which updates the conversation state.
4. The agent controller passes the updated state to the appropriate agent in the agent hub.
5. The agent decides on an action to take based on the current state.
6. The action is executed by the action framework.
7. The result of the action is passed back to the agent controller.
8. The agent controller updates the state and generates a response.
9. If speech output is enabled, the response is converted to speech by the TTS component.
10. The response is presented to the user through the UI.

## Extending the Architecture

The modular architecture of Aura makes it easy to extend and customize:

- **New Agents**: Add new agent implementations to the agent hub.
- **New Actions**: Add new action implementations to the action framework.
- **New Models**: Integrate new language models or speech processing models.
- **New UI Components**: Add new UI components for different interaction modes.