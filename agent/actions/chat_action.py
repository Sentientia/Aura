from action import Action

class ChatAction(Action):
    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def execute(self):
        pass