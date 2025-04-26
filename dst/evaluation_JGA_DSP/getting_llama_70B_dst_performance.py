import json
from tqdm import tqdm 

   



# ─────────────────────────────────────────────────────────────
# 1️⃣  UNIFIED MULTI-DOMAIN DST PROMPT  (response tag removed)
# ─────────────────────────────────────────────────────────────
DST_ALL_PROMPT = """
You are an AI Dialogue-State-Tracking (DST) assistant.

──────────────────────── TASK ────────────────────────
1. Read the full conversation history.
2. Fill/overwrite the slots for every domain the user mentions.

Domains & slots
───────────────
          
          ── Hotel slots ──
  pricerange   : {{expensive | cheap | moderate}}
  type         : {{guest house | hotel}}
  parking      : {{yes | no}}
  day          : {{monday | tuesday | … | sunday}}
  people       : {{1 | 2 | 3 | 4 | 5 | 6 | 7 | 8}}
  stay         : {{1 | 2 | 3 | 4 | 5 | 6 | 7 | 8}}
  internet     : {{yes | no}}
  name         : free text
  area         : {{centre | east | north | south | west}}
  star         : {{0 | 1 | 2 | 3 | 4 | 5}}

── Train slots ──
  arriveby    : 24‑h time (e.g. 06:00, 18:30)
  day         : {{monday | tuesday | wednesday | thursday | friday | saturday | sunday}}
  people      : {{1 | 2 | 3 | 4 | 5 | 6 | 7 | 8}}
  leaveat     : 24‑h time (e.g. 06:00, 18:30)
  destination : {{birmingham new street | bishops stortford | broxbourne | cambridge |
                  ely | kings lynn | leicester | london kings cross |
                  london liverpool street | norwich | peterborough |
                  stansted airport | stevenage}}
  departure   : same list as destination

── Restaurant slots ──
  pricerange : {{expensive | cheap | moderate}}
  area       : {{centre | east | north | south | west}}
  food       : free text
  name       : free text
  day        : {{monday | tuesday | wednesday | thursday | friday | saturday | sunday}}
  people     : {{1 | 2 | 3 | 4 | 5 | 6 | 7 | 8}}
  time       : 24‑h time (e.g. 06:00, 18:30)

── Attraction slots ──
  area : {{centre | east | north | south | west}}
  name : free text
  type : {{architecture | boat | cinema | college | concerthall | entertainment |
           museum | multiple sports | nightclub | park | swimmingpool | theatre}}

── Hospital slots ──
  department : free text

── Taxi slots ──
  leaveat     : 24‑h time (e.g. 06:00, 18:30)
  destination : free text
  departure   : free text
  arriveby    : 24‑h time (e.g. 06:00, 18:30)

── Profile slots ──
  profile_name
  profile_email
  profile_idnumber
  profile_phonenumber
  profile_platenumber     (all free text)




────────────────── OUTPUT FORMAT ──────────────────
Return **valid XML only**—nothing else.

<state>
  <!-- include a domain tag *only if* at least one slot is known -->
  <hotel>…</hotel>
  <train>…</train>
  <restaurant>…</restaurant>
  <attraction>…</attraction>
  <hospital>…</hospital>
  <taxi>…</taxi>
  <profile>…</profile>
</state>

Examples
────────
User: “I need a cheap hotel for 2 nights from Friday and a taxi to the Grand Arcade by 9 a.m.”
Assistant must output something like:

<hotel>
<pricerange>cheap</pricerange>
<stay>2</stay>
<day>friday</day>
</hotel>
<taxi>
<arriveby>09:00</arriveby>
<destination>Grand Arcade</destination>
</taxi>

"""



from openai import OpenAI
import re
import re

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8743/v1"

# completion = client.completions.create(model="meta-llama/Llama-3.3-70B-Instruct",
#                                       prompt="San Francisco is a")
# print("Completion result:", completion.choices[0].text)

def get_response(messages,model="meta-llama/Llama-3.3-70B-Instruct"):
    pass
    client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,)

    completion = client.chat.completions.create(model=model,messages=messages, temperature=0.1, max_tokens=200)

    return completion.choices[0].message.content.strip()

def get_history_as_strings(history):
    output_string = ""

    for message in history:
        role = message["role"]
        content = message["content"]
        output_string += f"{role}: {content}\n"
    
    return output_string



def parse_simple_xml(text: str) -> dict:
    """
    Grab all <tag>value</tag> pairs from `text`, even if they sit inside an
    outer wrapper (or have junk before/after), and return a dict.

    • Strings '', 'none', 'null'  →  None
    • Strings 'true', 'false'     →  bool
    • If an <output> tag is present, add key 'has_output' (True/False)
    """
    # ── 1.  Find *all* simple tag–value pairs  ──────────────────────────────
    matches = re.findall(r"<(\w+)>\s*(.*?)\s*</\1>", text, re.DOTALL | re.IGNORECASE)

    result = {}
    for tag, raw in matches:
        val = raw.strip()

        if val.lower() in {"", "none", "null","None", "NULL"}:
            parsed = None
        else:
            parsed = val

        result[tag] = parsed

    return result


def parse_simple_xml(text: str) -> dict:
    """
    Grab all <tag>value</tag> pairs from `text`, even if they sit inside an
    outer wrapper (or have junk before/after), and return a dict.

    • Strings '', 'none', 'null'  →  None
    • Strings 'true', 'false'     →  bool
    • If an <output> tag is present, add key 'has_output' (True/False)
    """
    # ── 1.  Find *all* simple tag–value pairs  ──────────────────────────────
    matches = re.findall(r"<(\w+)>\s*(.*?)\s*</\1>", text, re.DOTALL | re.IGNORECASE)

    result = {}
    for tag, raw in matches:
        val = raw.strip()

        if val.lower() in {"", "none", "null","None", "NULL"}:
            parsed = None
        else:
            parsed = val

        result[tag] = parsed

    return result


def main():
        
    PATH = '/Users/srijith/courses/Speech_project/Aura/data_for_llm/test_data.jsonl'
    save_path = '/Users/srijith/courses/Speech_project/Aura/data_for_llm/test_data_with_llama_70B_response.jsonl'
    with open(PATH) as f:
        # load jsonl 
        data = f.readlines()
        data = [json.loads(line) for line in data]
        # convert to json
        

    for num, datapoint in enumerate(tqdm(data)):

        input_text = datapoint['messages'][0]['content']
        input_text =  input_text.split('<INPUT_DIALOG>')[2].split('</INPUT_DIALOG>')[0]
        messages = [
                    {"role": "system", "content": DST_ALL_PROMPT},
                    {"role": "user", "content": input_text}
                ]
            
        
        llama_response = get_response(messages)
        datapoint['llama_response'] = llama_response

        if num % 100 == 0:
            with open(save_path, 'w') as f:
                for line in data:
                    f.write(json.dumps(line) + '\n')
            print("Saved at ", num)

    with open(save_path, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')
            