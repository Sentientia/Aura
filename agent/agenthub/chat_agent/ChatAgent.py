from agent.agenthub.base_agent import BaseAgent
from agent.controller.state import State
from agent.actions.action import Action
from agent.llm.openai_chat_completion import get_response
from agent.agenthub.chat_agent.prompts import get_prompt
from agent.actions.chat_action import ChatAction
import re
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
        action_observation_history = state.history
        dialog_state = state.dst
        prompt = get_prompt(action_observation_history, dialog_state)
        response = get_response(prompt)

        thought, action_type, payload = self.parse_response(response)
        print(f"\n*************INTERNAL-MONOLOGUE***************\nThought: {thought}, \nAction Type: {action_type}, \nPayload: {payload} \nDST state: {state.dst}\n****************INTERNAL-MONOLOGUE************\n")

        if action_type == "chat":
            action = ChatAction(thought=thought, payload=payload)
        return action
    
    def parse_response(self, response: str) -> tuple[str, str, str]:
        # Action will be within <action> tags
        # Payload will be within <payload> tags
        # Thought will be in <thought>
        # Use regex to find the tags
        thought_match = re.search(r'<thought>\s*(.*?)\s*</thought>', response, re.DOTALL)
        action_match = re.search(r'<action>\s*(.*?)\s*</action>', response, re.DOTALL)
        payload_match = re.search(r'<payload>\s*(.*?)\s*</payload>', response, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else "No thought provided"
        action = action_match.group(1).strip() if action_match else "chat"  # Default to chat if no action found
        payload = payload_match.group(1).strip() if payload_match else "No payload provided"

        return thought, action, payload
