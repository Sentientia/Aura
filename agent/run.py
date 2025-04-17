from pathlib import Path
wd = Path(__file__).parent.parent
import sys
sys.path.append(str(wd))

from controller.state import State
state = State()

while True:
    user_input = "Hi, Hello, How are you, LEANDER WANTS TO BOOK AN HOTEL FOR HIS FOREVER IMAGANERY GF" #input("You: ")
    user_input = {"role": "user", "content": user_input}
    state.history.append(user_input)
    output = state.get_tts_output()
    print("Assistant:", output)

