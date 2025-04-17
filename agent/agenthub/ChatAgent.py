from agent.agenthub.base_agent import BaseAgent
from agent.controller.state import State
from agent.actions.action import Action

class ChatAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.available_actions = [
            "chat",
            "search",
            "calendar",
            "finish"
        ]
    def step(self, state: State) -> Action:
        """
        Returns the next action to take based on the current state.
        """
        pass
