import json
from tqdm import tqdm 

PATH = 'data/text_5700_train_dev/data.json'
NAME_OUTPUT_TRAIN_FILE = 'train.jsonl'
NAME_OUTPUT_DEV_FILE = 'dev.jsonl'
NAME_OUTPUT_TEST_FILE = 'test.jsonl'
VAL_LIST_PATH = 'data/text_5700_train_dev/valListFile.json'
TEST_PATH = 'data/text_5700_test/data.json' 

PROMPT = """
You are an AI model performing Dialogue State Tracking (DST). Your task is to extract and update the dialogue state based on a conversation between a user and a system. 

The output dialog state is expected in json format enclosed within `<DST></DST>` tags. The keys and constraints for the output are provided in `<OUTPUT_FIELDS></OUTPUT_FIELDS>` tags. 
The actual dialog input between the user and assistant is provided in the `<INPUT_DIALOG></INPUT_DIALOG>` tags.

Your output should be a JSON object representing the dialogue state. The dialogue state contains only the following keys:  

['attraction', 'hospital', 'hotel', 'police', 'profile', 'restaurant', 'taxi', 'train'] 

Each of these parent keys can only contain specific child keys according to the following constraints:

<OUTPUT_FIELDS>
hotel:  
  - pricerange: price budget of the hotel; Possible values: ['expensive', 'cheap', 'moderate']  
  - type: type of the hotel; Possible values: ['guest house', 'hotel']  
  - parking: whether the hotel has parking; Possible values: ['no', 'yes']  
  - day: day of the hotel booking; Possible values: ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']  
  - people: number of people booking the hotel; Possible values: ['1', '2', '3', '4', '5', '6', '7', '8']  
  - stay: length of stay at the hotel; Possible values: ['1', '2', '3', '4', '5', '6', '7', '8']  
  - internet: whether the hotel has free internet; Possible values: ['no', 'yes']  
  - name: name of the hotel; Possible values: []  
  - area: area of the hotel; Possible values: ['centre', 'east', 'north', 'south', 'west']  
  - star: star of the hotel; Possible values: ['0', '1', '2', '3', '4', '5']  

train:  
  - arriveby: the arrival time of the train, 24-hour standard time, e.g. 06:00, 18:30; Possible values: []  
  - day: day of the train departure; Possible values: ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']  
  - people: number of people travelling by train; Possible values: ['1', '2', '3', '4', '5', '6', '7', '8']  
  - leaveat: leaving time of the train, 24-hour standard time, e.g. 06:00, 18:30; Possible values: []  
  - destination: destination of the train; Possible values: ['birmingham new street', 'bishops stortford', 'broxbourne', 'cambridge', 'ely', 'kings lynn', 'leicester', 'london kings cross', 'london liverpool street', 'norwich', 'peterborough', 'stansted airport', 'stevenage']  
  - departure: departure of the train; Possible values: ['birmingham new street', 'bishops stortford', 'broxbourne', 'cambridge', 'ely', 'kings lynn', 'leicester', 'london kings cross', 'london liverpool street', 'norwich', 'peterborough', 'stansted airport', 'stevenage']  

attraction:  
  - area: area of the attraction; Possible values: ['centre', 'east', 'north', 'south', 'west']  
  - name: name of the attraction; Possible values: []  
  - type: type of the attraction; Possible values: ['architecture', 'boat', 'cinema', 'college', 'concerthall', 'entertainment', 'museum', 'multiple sports', 'nightclub', 'park', 'swimmingpool', 'theatre']  

restaurant:  
  - pricerange: price budget for the restaurant; Possible values: ['expensive', 'cheap', 'moderate']  
  - area: area of the restaurant; Possible values: ['centre', 'east', 'north', 'south', 'west']  
  - food: the cuisine of the restaurant; Possible values: []  
  - name: name of the restaurant; Possible values: []  
  - day: day of the restaurant booking; Possible values: ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']  
  - people: number of people for the restaurant booking; Possible values: ['1', '2', '3', '4', '5', '6', '7', '8']  
  - time: time of the restaurant booking, 24-hour standard time, e.g. 06:00, 18:30; Possible values: []  

hospital:  
  - department: department of the hospital; Possible values: []  

taxi:  
  - leaveat: leaving time of taxi, 24-hour standard time, e.g. 06:00, 18:30; Possible values: []  
  - destination: destination of taxi; Possible values: []  
  - departure: departure location of taxi; Possible values: []  
  - arriveby: arrival time of taxi, 24-hour standard time, e.g. 06:00, 18:30; Possible values: []  

profile:  
  - name: the name of the user; Possible values: []  
  - email: the email of the user; Possible values: []  
  - idnumber: the idnumber of the user; Possible values: []  
  - phonenumber: the phonenumber of the user; Possible values: []  
  - platenumber: the platenumber of the user; Possible values: []  
</OUTPUT_FIELDS>
  

<INPUT_DIALOG>
{INPUT_TEXT}
</INPUT_DIALOG>
"""  


## Train and Validation Data Preparation

with open(VAL_LIST_PATH,'r') as f:
    valListFile = f.readlines()
valListFile = [x.strip() for x in valListFile]

with open(PATH) as f:
    data = json.load(f)

audio_clips = list(data.keys())

num_of_sig_clips = len([ i for i in audio_clips if 'SNG' in i])
num_of_multi_clips = len([ i for i in audio_clips if 'MUL' in i])
total_clips = len(audio_clips)

dev_sig_clips = int( .07 * num_of_sig_clips )
dev_multi_clips = int( .07 * num_of_multi_clips )
dev_total_clips = int( .07 * total_clips )


audio_clips = list(data.keys())

train_data = []
dev_data = []

for audio_clip in tqdm(audio_clips):
    audio_clip_data = data[audio_clip]
    dialog_state = audio_clip_data['goal']

    input_text = ''

    for turn in audio_clip_data['log']:
        text = turn['text']
        role = turn['tag']
        input_text +=  f"{role}: {text}\n"
        

    messages = [   {'role': 'user', 'content': PROMPT.format(INPUT_TEXT=input_text)}  ]
    messages.append({'role': 'assistant', 'content': f'<DST>\n{json.dumps(dialog_state)}\n</DST>'})
    jsonl_line_to_write = {'messages': messages }

    if 'MUL' in audio_clip and dev_multi_clips > 0:
        dev_multi_clips -= 1
        dev_data.append(jsonl_line_to_write)
    elif 'SNG' in audio_clip and dev_sig_clips > 0:
        dev_sig_clips -= 1
        dev_data.append(jsonl_line_to_write)
    else:
        train_data.append(jsonl_line_to_write)
        
with open(NAME_OUTPUT_TRAIN_FILE, 'w') as f:
    for line in train_data:
        f.write(json.dumps(line) + '\n')

with open(NAME_OUTPUT_DEV_FILE, 'w') as f:
    for line in dev_data:
        f.write(json.dumps(line) + '\n')


## Test Data Preparation

with open(TEST_PATH) as f:
    data = json.load(f)

audio_clips = list(data.keys())

test_data = []

for audio_clip in tqdm(audio_clips):
    audio_clip_data = data[audio_clip]
    dialog_state = audio_clip_data['goal']

    input_text = ''

    for turn in audio_clip_data['log']:
        text = turn['text']
        role = turn['tag']
        input_text +=  f"{role}: {text}\n"
        

    messages = [   {'role': 'user', 'content': PROMPT.format(INPUT_TEXT=input_text)}  ]
    messages.append({'role': 'assistant', 'content': f'<DST>\n{json.dumps(dialog_state)}\n</DST>'})
    jsonl_line_to_write = {'messages': messages }

    test_data.append(jsonl_line_to_write)

with open(NAME_OUTPUT_TEST_FILE, 'w') as f:
    for line in test_data:
        f.write(json.dumps(line) + '\n')