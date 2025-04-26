PATH = 'data_for_llm/original_test_data.json'
import json
from tqdm import tqdm
from getting_llama_70B_dst_performance import get_response, DST_ALL_PROMPT

def flatten_dict(d, parent_key='', sep='-'):
    """ Recursively flattens a nested dictionary. """
    flat_dict = {}
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k  # Concatenate keys
        
        if isinstance(v, dict):  # If value is a dict, recurse
            flat_dict.update(flatten_dict(v, new_key, sep))
        else:
            flat_dict[new_key] = v  # Store final key-value pair
            
    return flat_dict


with open(PATH) as f:
    data = json.load(f)


#lets choose equal single and multi clips to test llm inference
multi_clips = 100
single_clips = 100

test_data = {}
for audio_file_name in data:
    if multi_clips > 0 and 'MUL' in audio_file_name:
        multi_clips -= 1
        test_data[audio_file_name] = data[audio_file_name]
    elif single_clips > 0 and 'SNG' in audio_file_name:
        single_clips -= 1
        test_data[audio_file_name] = data[audio_file_name]


counter = 0
save_path = 'data_for_llm/save_for_jpa'
for num ,audio_file_name in enumerate(tqdm(test_data)):
    datapoint = test_data[audio_file_name]
    logs = datapoint['log']


    cumulative_text = ''
    prev_turn__metadata = {}
    prev_llama_prediction = {}
    for turn_index, turn in enumerate(logs):
        cumulative_text += turn['tag'] + ': ' + turn['text'] + '\n'
        metadata = turn['metadata']
        metadata = flatten_dict(metadata)
        metadata = {k: v for k, v in metadata.items() if v != '' and v != []}

        if turn['tag'] == 'system' and  len(metadata) != len(prev_turn__metadata) :
            counter += 1
            messages = [{"role": "system", "content": DST_ALL_PROMPT},{"role": "user", "content": cumulative_text} ]
            turn['llama_response'] =   get_response(messages)
            prev_llama_prediction = turn['llama_response']
            turn['ran_llama_inference'] = True    

            prev_turn__metadata = metadata
        elif turn['tag'] == 'system' and  len(metadata) == len(prev_turn__metadata) :
            turn['llama_response'] = prev_llama_prediction
            turn['ran_llama_inference'] = False
        # print(f'{num}: {audio_file_name} turn {turn_index}{turn["tag"]} metadata: {metadata}')
        # print(cumulative_text)
        # print(metadata)
        # print("--------------------------------------------------")

    test_data[audio_file_name]['done'] = True

    if num % 20 == 0 :
        temp_save_path = save_path+'/'+ str(num) + '.json'
        with open(temp_save_path, 'w') as f:
            json.dump(test_data, f,indent=1)
        print("Saved at ", num)

temp_save_path = save_path+'/final'+ + '.json'
with open(temp_save_path, 'w') as f:
    json.dump(test_data, f,indent=1)
print("Saved at ", num)


counter = 0