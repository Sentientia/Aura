from pathlib import Path
wd = Path(__file__).parent.parent.parent
import sys
sys.path.append(str(wd))

from agent.controller.state import State
from agent.agenthub.chat_agent.ChatAgent import ChatAgent

state = State()
agent = ChatAgent()
agent.step(state)

while True:
    action = agent.step(state)
    observation = action.execute()
    state.history.append({'action': action.payload, 'observation': observation}) #Thought may be added later

