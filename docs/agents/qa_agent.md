# QA Agent

The QA Agent is a specialized agent for question-answering tasks. It is designed to answer specific questions from users, using web search when necessary to find the most up-to-date information.

## Overview

The QA Agent is implemented in the `QAAgent` class, which extends the `BaseAgent` class. It is designed to work in the QA Evaluation mode, where it answers specific questions without engaging in extended conversations.

## Capabilities

The QA Agent can perform the following actions:

- **Answer**: Provide a direct answer to the user's question.
- **Web Search**: Search the web for information to help answer the question.

## Implementation

The QA Agent is implemented in the `agent/agenthub/qa_agent/QAAgent.py` file. It uses the following components:

- **Prompt Template**: Defined in `agent/agenthub/qa_agent/qa_prompt.py`.
- **LLM Integration**: Uses the OpenAI Chat Completion API through `agent/llm/openai_chat_completion.py`.
- **Action Framework**: Uses the action classes defined in `agent/actions/`.

## Usage

The QA Agent is used in the QA Evaluation mode. To use the QA Agent:

1. Initialize the controller in QA Evaluation mode:
   ```python
   from agent.controller.controller import Controller
   from agent.controller.modes import Mode

   controller = Controller(operation_mode=Mode.QA_EVAL)
   ```

2. Run the QA evaluation with a specific question:
   ```python
   history, instructions = controller.qa_eval({
       "instruction": "What is the capital of France?",
       "additional_instruction": None
   })
   ```

3. The history will contain the trajectory of actions and observations, and the final answer will be in the last observation.

## Prompt Format

The QA Agent uses a structured prompt format to communicate with the language model. The prompt includes:

- **System Prompt**: Defines the agent's role, capabilities, and response format.
- **Question**: The specific question to be answered.
- **Action-Observation History**: The history of actions taken by the agent and the observations from those actions.
- **Instructions**: Any additional instructions for the agent, such as whether to answer in the current step or continue searching.

The agent's response is expected to include:

- **Thought**: The agent's reasoning about the question and how to answer it.
- **Action**: The type of action to take (answer, web_search).
- **Payload**: The content of the action, such as the answer to the question or the search query to execute.

## Example

Here's an example of how the QA Agent processes a question:

1. Question: "What is the capital of France?"

2. Agent thought: "I know that the capital of France is Paris."

3. Agent action: "answer"

4. Agent payload: "The capital of France is Paris."

For more complex questions that require up-to-date information:

1. Question: "Who is the current president of the United States?"

2. Agent thought: "This is a time-sensitive question, so I should use web search to get the most up-to-date information."

3. Agent action: "web_search"

4. Agent payload: {"google_search_query": "who is the current president of the United States", "wikipedia_search_query": "United States President"}

5. After receiving the search results, the agent processes them and provides the answer:

6. Agent thought: "Based on the search results, I can see that the current president of the United States is [Name]."

7. Agent action: "answer"

8. Agent payload: "The current president of the United States is [Name]."