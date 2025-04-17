class Action:
    def __init__(self, thought: str = '', payload: dict[str, any] = {}):
        self.thought = thought
        self.payload = payload


    def execute(self):
        pass
