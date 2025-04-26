PATH = 'data_for_llm/save_for_jpa/final.json'
from error_computation import * 

import json
import numpy as np
import re
from fuzzywuzzy import process
import os

"-------------------------------HARD CODED PATHS-------------------------------"
ONTOLOGY_PATH = 'data_for_llm/ontology.json'
with open(ONTOLOGY_PATH) as f:
    ontology = json.load(f)
"------------------------------------------------------------------------------"

with open(PATH) as f:
    data = json.load(f)


error_logging_dict = {}
score_for_data = []

for audio_clip_name in data:
    # if "MUL" not in audio_clip_name:
    #     continue
    datapoint = data[audio_clip_name]
    logs = datapoint['log']

    score_for_audio_clip = []
    input_text = ''
    for turn_index, turn in enumerate(logs):
        tag = turn['tag']
        input_text += tag + ': ' + turn['text'] + '\n'

        if tag != 'system':
            continue

        metadata = turn['metadata']
        metadata = flatten_dict(metadata)
        metadata = {k: v for k, v in metadata.items() if v != '' and v != []}
        metadata = transform_keys(metadata)

        predict = turn['llama_response']
        predict = parse_simple_xml_recursive(predict) if predict != {} else {}
        if 'state' in predict.keys():
            predict = predict['state']
        if predict == None:
            predict = {}
        predict = flatten_dict(predict)
        predict = transform_keys(predict)

        exact_match = 1.0
        for metadata_keys in metadata.keys():
            if metadata[metadata_keys] != predict.get(metadata_keys, {'a':0}):
                exact_match = 0.0
                if metadata_keys not in error_logging_dict:
                    error_logging_dict[metadata_keys] = []
                error_logging_dict[metadata_keys].append({ 
                    'audio_clip_name': audio_clip_name,
                    'turn_index': turn_index,
                    'metadata': metadata[metadata_keys],
                    'prediction': predict.get(metadata_keys, "missed"),})

        score_for_audio_clip.append(exact_match)

    score_for_data.append(np.mean(score_for_audio_clip))
    
   # print(f"Audio clip {audio_clip_name} score: {np.mean(score_for_audio)}")
print(f"Average score for all audio clips: {np.mean(score_for_data)}")
 

        


