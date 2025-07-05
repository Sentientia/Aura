# Agents Overview

Aura provides a flexible agent framework that allows for different agent implementations to be used for different use cases. This page provides an overview of the available agents and how they work.

## Agent Framework

The agent framework is built around the concept of a base agent that provides common functionality, with specific agent implementations extending this base agent to provide specialized behavior.

### Base Agent

The base agent provides common functionality such as:

- State management
- Action selection
- Response generation

All specific agent implementations extend this base agent.

## Available Agents

Aura currently provides the following agent implementations:

### [Chat Agent](chat_agent.md)

The Chat Agent is a general-purpose agent for conversational interactions. It is designed to engage in natural conversations with users, understand their requests, and take appropriate actions to fulfill those requests.

### [QA Agent](qa_agent.md)

The QA Agent is a specialized agent for question-answering tasks. It is designed to answer specific questions from users, using web search when necessary to find the most up-to-date information.

## Agent Selection

The appropriate agent is selected based on the mode of operation:

- **UI Mode**: Uses the Chat Agent for interactive conversations.
- **QA Evaluation Mode**: Uses the QA Agent for question-answering tasks.

## Extending with New Agents

The agent framework is designed to be extensible, allowing for new agent implementations to be added as needed. To create a new agent:

1. Create a new class that extends the `BaseAgent` class.
2. Implement the `step` method to define the agent's behavior.
3. Register the agent in the controller for the appropriate mode of operation.

For more details on creating new agents, see the [Contributing](../contributing.md) guide.