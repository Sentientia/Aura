from pathlib import Path
wd = Path(__file__).parent.parent
import sys
sys.path.append(str(wd))

from controller.state import State
state = State()

test_hotel_messages = [
    {"role": "user", "content": "Hi"},
    {
        "role": "assistant",
        "content": "Hello! How can I help you with your hotel booking today?"
    },
    {
        "role": "user",
        "content": "I'm looking for a cheap hotel in the city centre."
    },
    {
        "role": "assistant",
        "content": "Great! What day will you check in, and how many nights would you like to stay?"
    },
    {
        "role": "user",
        "content": "I'll arrive on Friday and stay for 3 nights."
    },
    {
        "role": "assistant",
        "content": "Got it. How many people will be staying?"
    },
    {
        "role": "user",
        "content": "Two people."
    },
    {
        "role": "assistant",
        "content": "Do you need parking and free internet?"
    },
    {
        "role": "user",
        "content": "Yes, I'd like both parking and free internet."
    },
    {
        "role": "assistant",
        "content": "Do you have a preferred star rating or a specific hotel in mind?"
    },
    {
        "role": "user",
        "content": "A 3‑star place would be nice. Do you have Hamilton Lodge available?"
    },
    {
        "role": "assistant",
        "content": "Hamilton Lodge, 3‑star, cheap, centre area, with parking and free internet—perfect. May I have a name for the booking and your email address?"
    },
    {
        "role": "user",
        "content": "Sure, the name is Srijith, and my email is srijith@example.com."
    },
    {
        "role": "assistant",
        "content": (
            "Thanks, Srijith! Here’s a summary of your request:\n"
            "• Hotel: Hamilton Lodge (3‑star, cheap)\n"
            "• Area: centre\n"
            "• Check‑in: Friday, 3‑night stay\n"
            "• Guests: 2\n"
            "• Parking: yes\n"
            "• Internet: yes\n\n"
            "I’ll confirm the booking and email you the details shortly."
        )
    }
]

for i in test_hotel_messages:
    #user_input = i  #input("You: ")
    #print("User:", user_input)
    #user_input = {"role": "user", "content": user_input}
    #state.history.append(user_input)
    state.history.append(i)
    output = state.get_tts_output()
    print(f"{i['role'].capitalize()}:", i['content'])
    print("DST:", output)
    #print("Assistant:", output)

