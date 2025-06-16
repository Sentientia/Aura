from agent.actions.action import Action
from agent.controller.state import State

class AnswerAction(Action):
    def __init__(self, thought: str, payload: str):
        super().__init__(thought, payload)

    def execute(self, state: State) -> str:
        state.history.append({"action": {"type":"answer", "payload":self.payload}, "observation": {"type":"answer", "payload":"Trajectory Ended"}})
        return self.payload