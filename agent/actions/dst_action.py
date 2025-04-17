from actions.action import Action
from llm.openai_chat_completion import get_response

class DSTAction(Action):
    def __init__(self, thought: str = '', history: dict[str, any] = {}):
        super().__init__()
        #self.message = message

    def execute(self,type: str = ''):
        
        if type == 'get_category':
            PROMPT = """
You are an AI model performing Dialogue State Tracking (DST). Your task is to extract and update the dialogue state based on a conversation between a user and a system. 

To begin with you are supposed to predict which category the user input belongs to within the following categories:

'attraction': The user wants to vist a place of interest and you can provide information about the following tasks : 'architecture', 'boat', 'cinema', 'college', 'concerthall', 'entertainment', 'museum', 'multiple sports', 'nightclub', 'park', 'swimmingpool', 'theatre'
'hotel' : The user wants to book a hotel 
'train' : The user wants to book a train
'restaurant' : The user wants to book a restaurant
'hospital' : The user wants to book a hospital
'taxi' : The user wants to book a taxi
'profile' : information about the user

You have to output which category the user input belongs to. If the user is just greeting return a friendly greeting and how you can help

"""  



        prompt = f"""Below is  a conversation history between a user and an agent. Your job is to figure out what the user is tryin
The user may start with a greeting or be trying to """
        pass

