# Web Search Action

The Web Search Action is used to search the web for information. It can be used to find answers to questions, get up-to-date information, or research topics.

## Overview

The Web Search Action is implemented in the `WebSearchAdvancedAction` class, which extends the `Action` class. It is designed to perform web searches and return the results to the agent.

## Capabilities

The Web Search Action can:

- Perform Google searches
- Perform Wikipedia searches
- Process and format search results
- Handle different types of search queries

## Implementation

The Web Search Action is implemented in the `agent/actions/web_search_advanced_action.py` file. It uses the following components:

- **Action Base Class**: Extends the `Action` class defined in `agent/actions/action.py`.
- **Search API Integration**: Uses external search APIs to perform searches.
- **Result Processing**: Processes and formats search results for the agent.

## Usage

The Web Search Action is used by the agent to find information on the web. To use the Web Search Action:

1. Create a new instance of the `WebSearchAdvancedAction` class with the appropriate thought and payload:
   ```python
   from agent.actions.web_search_advanced_action import WebSearchAdvancedAction

   action = WebSearchAdvancedAction(
       thought="I should search for information about Italian restaurants",
       payload={"google_search_query": "best Italian restaurants downtown", "wikipedia_search_query": "Italian cuisine"}
   )
   ```

2. Execute the action with the current state:
   ```python
   observation = action.execute(state)
   ```

3. The observation will contain the search results.

## Example

Here's an example of how the Web Search Action is used to find information:

1. Agent creates a Web Search Action:
   ```python
   action = WebSearchAdvancedAction(
       thought="I should search for information about the weather",
       payload={"google_search_query": "weather forecast today", "wikipedia_search_query": "weather"}
   )
   ```

2. Agent executes the action:
   ```python
   observation = action.execute(state)
   ```

3. The action performs a Google search for "weather forecast today" and a Wikipedia search for "weather".

4. The observation contains the search results, which might include:
   - Current weather conditions
   - Weather forecast for the day
   - General information about weather patterns
   - Links to weather-related resources

5. The agent processes the observation and creates a new action based on the search results, typically a Chat Action to present the information to the user.

## Integration with Other Actions

The Web Search Action is often used in conjunction with other actions to provide a complete interaction flow. For example:

1. User asks about the weather.
2. Agent uses a Web Search Action to find weather information.
3. Agent processes the search results.
4. Agent uses a Chat Action to present the weather information to the user.

This combination of actions allows the agent to provide accurate and up-to-date information to the user.