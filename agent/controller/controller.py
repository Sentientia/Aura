from pathlib import Path
wd = Path(__file__).parent.parent.parent
import sys
sys.path.append(str(wd))

from agent.controller.state import State
from agent.agenthub.chat_agent.ChatAgent import ChatAgent
from agent.dst.dst_action import DSTAction
from agent.actions.chat_action import ChatAction

state = State()
agent = ChatAgent()
dst = DSTAction()

while True:
    action = agent.step(state)
    observation = action.execute()

    if isinstance(action, ChatAction):
        state.conversation.extend([{"role": "assistant", "content": action.payload},{"role": "user", "content": observation}])

    state.history.extend([{"role": "assistant", "content": action.payload},{"role": "user", "content": observation}])
     

    if state.dst_class is None:
        state.dst_class = DSTAction(thought='Have to get DST category', payload=state.conversation).execute(type='get_category')['output']
    else:
        state.dst = DSTAction(thought='Have to get DST category', payload=state.conversation).execute(type=state.dst_class)
    



        
            #     self.dst_class = None
            #     #self.history.append({"role": "assistant", "content": initial_dst_classify['response']})
            #     return initial_dst_classify['response']
            # else:
            #     self.dst_class = initial_dst_classify['output']
            #     print("***DST Class set to :", self.dst_class,'***')

    

