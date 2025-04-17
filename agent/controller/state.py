from agent.actions.dst_action import DSTAction
class State:
    def __init__(self):
        self.history = []
        self.dst = None

       
    def get_state(self):
        pass

    def get_tts_output(self):

        if self.dst is None:
            self.dst = DSTAction(thought='Have to get DST', payload=self.history).execute(type='get_category')
            # Have to figure out which category the user input belongs to to ask relavent questions

        
