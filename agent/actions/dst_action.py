from agent.actions.action import Action
from agent.llm.openai_chat_completion import get_response, get_history_as_strings

class DSTAction(Action):
    def __init__(self, thought: str = '', payload: dict[str, any] = {}):
        super().__init__(thought, payload)
        #self.message = message

    def execute(self,type: str = ''):
        
        if type == 'get_category':
            PROMPT ="""
You are an AI model performing Dialogue State Tracking (DST). Your task is to extract and update the dialogue state based on a conversation between a user and a system. 

To begin with you are supposed to predict which category the user input belongs to within the following categories:

'attraction': The user wants to vist a place of interest and you can provide information about the following tasks : 'architecture', 'boat', 'cinema', 'college', 'concerthall', 'entertainment', 'museum', 'multiple sports', 'nightclub', 'park', 'swimmingpool', 'theatre'
'hotel' : The user wants to book a hotel 
'train' : The user wants to book a train
'restaurant' : The user wants to book a restaurant
'hospital' : The user wants to book a hospital
'taxi' : The user wants to book a taxi
'profile' : information about the user

You have to output which category the user input belongs to. If the user is just greeting return a friendly greeting and retun None

You  have to retrun the answer as in JSON format so that python code can directly read out the output , eg: 
output:None 

Below is the user converstaion 

{USER_HISTORY}
"""

            PROMPT = """
You are an AI Dialogue State‑Tracking (DST) assistant.

───────────  TASK  ───────────
1. Read the latest user message.
2. Decide which single intent best matches it:

   • attraction     – user wants to visit a place (sub‑types: architecture, boat,
                      cinema, college, concerthall, entertainment, museum,
                      multiple_sports, nightclub, park, swimmingpool, theatre)
   • hotel          – user wants to book / ask about a hotel
   • train          – user wants to book / ask about a train
   • restaurant     – user wants to find / book a restaurant
   • hospital       – user seeks medical help / info
   • taxi           – user wants to book a taxi
   • profile        – user is giving personal info about themselves

3. If the user message is ONLY a greeting (examples: “hi”, “hello”, “good morning”),
   reply with a friendly greeting and leave <output> empty.

────────  OUTPUT  FORMAT  ────────
Return **valid XML** exactly like the examples below—nothing else.

Example 1 – category identified:
<dst>
  <response></response>
  <output>hotel</output>
</dst>

Example 2 – simple greeting:
<dst>
  <response>Hello! How can I help you today?</response>
  <output></output>
</dst>

User message:
{USER_HISTORY}
"""

            
            user_history = get_history_as_strings(self.payload)
            PROMPT = PROMPT.format(USER_HISTORY=user_history)

            input =[ {"role": "system", "content": f"{PROMPT}"}]
            output = get_response(PROMPT)
           



