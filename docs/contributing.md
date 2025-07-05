# Contributing

Thank you for your interest in contributing to Aura! This guide will help you get started with contributing to the project.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/Aura.git
   cd Aura
   ```
3. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Make your changes.
5. Commit your changes:
   ```bash
   git commit -m "Add your commit message here"
   ```
6. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. Create a pull request on GitHub.

## Development Environment

Follow the [Installation](installation.md) guide to set up your development environment.

## Project Structure

Familiarize yourself with the [project structure](index.md#repository-structure) before making changes.

## Adding New Agents

To add a new agent:

1. Create a new directory in `agent/agenthub/` for your agent.
2. Create a new class that extends the `BaseAgent` class.
3. Implement the `step` method to define the agent's behavior.
4. Register the agent in the controller for the appropriate mode of operation.

Example:

```python
from agent.agenthub.base_agent import BaseAgent
from agent.controller.state import State
from agent.actions.action import Action

class MyNewAgent(BaseAgent):
    def __init__(self, mode=Mode.UI, io_mode=Mode.TEXT_2_TEXT_CASCADED):
        super().__init__()
        self.mode = mode
        self.io_mode = io_mode
    
    def step(self, state: State) -> Action:
        # Implement your agent's behavior here
        pass
```

## Adding New Actions

To add a new action:

1. Create a new file in `agent/actions/` for your action.
2. Create a new class that extends the `Action` class.
3. Implement the `execute` method to define the action's behavior.
4. Register the action in the agent's `step` method.

Example:

```python
from agent.actions.action import Action
from agent.controller.state import State

class MyNewAction(Action):
    def __init__(self, thought=None, payload=None):
        super().__init__(thought, payload)
    
    def execute(self, state: State):
        # Implement your action's behavior here
        pass
```

## Documentation

When adding new features or making changes, please update the documentation accordingly:

1. Update the relevant Markdown files in the `docs/` directory.
2. Add new Markdown files for new features if necessary.
3. Update the `mkdocs.yml` file to include new documentation pages.

## Testing

Before submitting a pull request, please test your changes:

1. Run the existing tests:
   ```bash
   python -m unittest discover
   ```
2. Add new tests for your changes if necessary.

## Code Style

Please follow the existing code style:

- Use 4 spaces for indentation.
- Use descriptive variable and function names.
- Add docstrings to classes and functions.
- Follow PEP 8 guidelines.

## Pull Request Process

1. Ensure your code follows the code style guidelines.
2. Update the documentation if necessary.
3. Add tests for your changes if necessary.
4. Ensure all tests pass.
5. Submit your pull request with a clear description of the changes.

## License

By contributing to Aura, you agree that your contributions will be licensed under the project's license.