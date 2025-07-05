# Actions Overview

Aura provides a flexible action framework that allows agents to perform various tasks to fulfill user requests. This page provides an overview of the available actions and how they work.

## Action Framework

The action framework is built around the concept of a base action that provides common functionality, with specific action implementations extending this base action to provide specialized behavior.

### Base Action

The base action provides common functionality such as:

- Execution logic
- State management
- Result formatting

All specific action implementations extend this base action.

## Available Actions

Aura currently provides the following action implementations:

### [Chat Action](chat_action.md)

The Chat Action is used to engage in conversation with the user. It is the most basic action and is used for all text-based interactions.

### [Web Search Action](web_search_action.md)

The Web Search Action is used to search the web for information. It can be used to find answers to questions, get up-to-date information, or research topics.

### [Calendar Action](calendar_action.md)

The Calendar Action is used to manage calendar events. It can create, read, update, and delete events on the user's calendar.

### [Email Action](email_action.md)

The Email Action is used to send and manage emails. It can compose and send emails on behalf of the user.

### [Contact Action](contact_action.md)

The Contact Action is used to manage contact information. It can retrieve contact details from the user's address book.

### [Answer Action](answer_action.md)

The Answer Action is used by the QA Agent to provide direct answers to questions. It is a specialized action for question-answering tasks.

## Action Selection

The appropriate action is selected by the agent based on the current state of the conversation and the user's request. The agent uses its reasoning capabilities to determine which action is most appropriate for the current situation.

## Extending with New Actions

The action framework is designed to be extensible, allowing for new action implementations to be added as needed. To create a new action:

1. Create a new class that extends the `Action` class.
2. Implement the `execute` method to define the action's behavior.
3. Register the action in the agent's `step` method.

For more details on creating new actions, see the [Contributing](../contributing.md) guide.