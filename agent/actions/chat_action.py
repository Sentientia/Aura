from agent.actions.action import Action

class ChatAction(Action):
    def __init__(self, thought: str, payload: str):
        super().__init__(thought, payload)

    def execute(self) -> str:
        print(self.payload)
        return str(input('Enter your response: '))