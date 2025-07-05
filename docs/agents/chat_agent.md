# Chat Agent

The Chat Agent is a general-purpose agent for conversational interactions. It is designed to engage in natural conversations with users, understand their requests, and take appropriate actions to fulfill those requests.

## Overview

The Chat Agent is implemented in the `ChatAgent` class, which extends the `BaseAgent` class. It is designed to work in the UI mode, where it interacts with users through a conversational interface.

## Capabilities

The Chat Agent can perform the following actions:

- **Chat**: Engage in conversation with the user.
- **Web Search**: Search the web for information.
- **Calendar**: Manage calendar events.
- **Contact**: Manage contact information.
- **Email**: Send and manage emails.

## Implementation

The Chat Agent is implemented in the `agent/agenthub/chat_agent/ChatAgent.py` file. It uses the following components:

- **Prompt Template**: Defined in `agent/agenthub/chat_agent/prompts.py`.
- **LLM Integration**: Uses the OpenAI Chat Completion API through `agent/llm/openai_chat_completion.py`.
- **Action Framework**: Uses the action classes defined in `agent/actions/`.

## Usage

The Chat Agent is used in the UI mode, which is the default mode of operation for the Aura system. To use the Chat Agent:

1. Initialize the controller in UI mode:
   ```python
   from agent.controller.controller import Controller
   from agent.controller.modes import Mode

   controller = Controller(operation_mode=Mode.UI)
   ```

2. Add user input to the state:
   ```python
   controller.add_user_input("Hello, how can you help me?")
   ```

3. Get the next action and observation:
   ```python
   action, observation = controller.get_next_chat_action()
   ```

4. The action will be a `ChatAction` or another action type, and the observation will be the result of executing that action.

## Prompt Format

The Chat Agent uses a structured prompt format to communicate with the language model. The prompt includes:

- **System Prompt**: Defines the agent's role, capabilities, and response format.
- **Conversation History**: The history of the conversation between the user and the agent.
- **Dialog State**: The current state of the dialog, including information gathered from the user.
- **Action-Observation History**: The history of actions taken by the agent and the observations from those actions.

The agent's response is expected to include:

- **Thought**: The agent's reasoning about the current state and what action to take.
- **Action**: The type of action to take (chat, web_search, calendar, contact, email).
- **Payload**: The content of the action, such as the message to send to the user or the search query to execute.

## Example

Here's an example of how the Chat Agent processes a user request:

1. User input: "I need to find a good Italian restaurant in downtown."

2. Agent thought: "The user is looking for restaurant recommendations. I should ask for more specific information about their preferences."

3. Agent action: "chat"

4. Agent payload: "I'd be happy to help you find an Italian restaurant in downtown. Do you have any specific preferences for price range or type of Italian cuisine?"

5. User input: "I'm looking for a mid-range restaurant with good pasta dishes."

6. Agent thought: "Now I have more specific information. I should search for Italian restaurants in downtown that match these criteria."

7. Agent action: "web_search"

8. Agent payload: {"google_search_query": "mid-range Italian restaurant downtown good pasta dishes", "wikipedia_search_query": "Italian restaurants downtown"}

9. After receiving the search results, the agent processes them and responds to the user with recommendations.