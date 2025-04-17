class Action:
    def __init__(self, thought: str = '', payload: dict[str, any] = {}):
        self.thought = thought
        self.history = payload
        pass


    def execute(self):
        pass


        
