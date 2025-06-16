from agent.agenthub.base_agent import BaseAgent
from agent.controller.state import State
from agent.actions.action import Action
from agent.llm.openai_chat_completion import get_response
from agent.actions.web_search_advanced_action import WebSearchAdvancedAction
from agent.actions.answer_action import AnswerAction
import re
from agent.controller.modes import Mode
from agent.agenthub.qa_agent.qa_prompt import get_qa_prompt

class QAAgent(BaseAgent):
    def __init__(self, mode:Mode=Mode.QA_EVAL, io_mode:Mode=Mode.SPEECH_2_TEXT_CASCADED):
        super().__init__()
        self.mode = mode
        self.io_mode = io_mode
       
    def step(self, state: State) -> Action:
        """
        Returns the next action to take based on the current state.
        """ 

        prompt = get_qa_prompt(state.special_instructions, state.history, state.terminate_trajectory)
        response = get_response(prompt)

        thought, action_type, payload = self.parse_response(response)
        # print(f"\n*************INTERNAL-MONOLOGUE***************\nThought: {thought}, \nAction Type: {action_type}, \nPayload: {payload} \nConversation History: {state.conversation} \nAction-Observation History: {state.history} \nDST state: {state.dst}\n****************INTERNAL-MONOLOGUE************\n")

        if action_type == "answer":
            action = AnswerAction(thought=thought, payload=payload)
        elif action_type == "web_search":
            action = WebSearchAdvancedAction(thought=thought, payload=payload)
        return action
    
    def parse_response(self, response: str) -> tuple[str, str, str]:
        # Action will be within <action> tags
        # Payload will be within <payload> tags
        # Thought will be in <thought>
        thought_match = re.search(r'<thought>\s*(.*?)\s*</thought>', response, re.DOTALL)
        action_match = re.search(r'<action>\s*(.*?)\s*</action>', response, re.DOTALL)
        payload_match = re.search(r'<payload>\s*(.*?)\s*</payload>', response, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else "No thought provided"
        action = action_match.group(1).strip() if action_match else "answer"  # Default to chat if no action found
        payload = payload_match.group(1).strip() if payload_match else "No payload provided"

        return thought, action, payload
