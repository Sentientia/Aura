from pathlib import Path
from agent.controller.modes import Mode
wd = Path(__file__).parent.parent.parent
import sys
sys.path.append(str(wd))

from agent.controller.state import State
from agent.agenthub.chat_agent.ChatAgent import ChatAgent
from agent.dst.dst_action import DSTAction
from agent.actions.chat_action import ChatAction
from agent.speech_utils.utils import get_transcript
from agent.agenthub.qa_agent.QAAgent import QAAgent
from agent.actions.answer_action import AnswerAction

class Controller:
    def __init__(self, operation_mode:Mode=Mode.UI, io_mode=Mode.TEXT_2_TEXT_CASCADED, max_iterations:int=3):
        self.state = State()
        self.agent = QAAgent(mode=operation_mode, io_mode=io_mode)
        if operation_mode == Mode.UI:
            self.agent = ChatAgent(mode=operation_mode, io_mode=io_mode)
        elif operation_mode == Mode.QA_EVAL:
            self.agent = QAAgent(mode=operation_mode, io_mode=io_mode)
        self.dst = DSTAction()
        self.operation_mode = operation_mode
        self.max_iterations = max_iterations
        self.io_mode = io_mode
        self.eval_map = {
            Mode.QA_EVAL: self.qa_eval
        }

    def reset(self): #TODO: Rethink this because max_iterations may need to be reset?
        self.state = State()
        self.dst = DSTAction()

    def get_next_chat_action(self): #Used for UI mode with T2T
        action = None
        observation = None
        while not isinstance(action, ChatAction):
            action = self.agent.step(self.state)
            observation = action.execute(self.state)

        return action, observation
    
    def add_user_input(self, user_input): #Used for UI mode with T2T
        self.state.conversation.append({"role": "user", "content": user_input})
        if len(self.state.history) > 0 and self.state.history[-1]['observation']['payload'] is None:
            self.state.history[-1]['observation']['payload'] = user_input
        elif len(self.state.history) == 0:
            self.state.history.append({"action": {"type":None, "payload":None} , 
                              "observation": {"type":"chat", "role":"user", "payload":user_input}})
    
    def qa_eval(self,input:dict)->list[dict]:
        """
        input: {
            instruction:[(audio,sr)|text]
            additional_instruction:str
        }
        Returns:
            Trajectory history
        """
        #TODO: Handle E2E separately
        if self.io_mode == Mode.SPEECH_2_TEXT_CASCADED:
            audio, sr = input['instruction']
            instruction = get_transcript(audio, sr, "Whisper v3 Large")
            print(f"Transcript: {instruction}")
        elif self.io_mode == Mode.TEXT_2_TEXT_CASCADED:
            instruction = input['instruction']
        else:
            raise ValueError(f"Invalid IO mode or IO Mode not implemented: {self.io_mode}")
        
        if 'additional_instruction' in input and input['additional_instruction'] is not None:
            instruction += f"\n{input['additional_instruction']}"
        
        self.state.special_instructions = instruction
        
        iterations = 1
        while iterations <= self.max_iterations:
            if iterations == self.max_iterations:
                self.state.terminate_trajectory = True
            action = self.agent.step(self.state)
            observation = action.execute(self.state)
            if isinstance(action, AnswerAction):
                break
            iterations += 1
        
        return self.state.history, self.state.special_instructions
    
    def run(self):
        while True:
            action = self.agent.step(self.state)
            observation = action.execute()

            if isinstance(action, ChatAction):
                self.state.conversation.extend([{"role": "assistant", "content": action.payload},{"role": "user", "content": observation}])

            self.state.history.extend([{"role": "assistant", "content": action.payload},{"role": "user", "content": observation}])
            

            if self.state.dst_class is None:
                self.state.dst_class = DSTAction(thought='Have to get DST category', payload=state.conversation).execute(type='get_category')['output']
            else:
                self.state.dst = DSTAction(thought='Have to get DST category', payload=self.state.conversation).execute(type=self.state.dst_class)
    



        
            #     self.dst_class = None
            #     #self.history.append({"role": "assistant", "content": initial_dst_classify['response']})
            #     return initial_dst_classify['response']
            # else:
            #     self.dst_class = initial_dst_classify['output']
            #     print("***DST Class set to :", self.dst_class,'***')

    

