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

def extract_json_from_tag(tag, data):

    match = re.search(r'<DST>\n(.*?)\n</DST>', data, re.DOTALL)
    if match:
        try:
            extracted_json = match.group(1)
            extracted_dict = json.loads(extracted_json)
            # print(extracted_dict) 
        except json.JSONDecodeError:
            print(f"Error decoding JSON from <{tag}> tag: {extracted_json}")
            extracted_dict = {}
    else:
        print("No valid JSON found inside <DST> tags.")
        extracted_dict = {}
    return extracted_dict


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

def transform_keys(d, sep='-'):
    """Modifies dictionary keys by removing the middle part if they have three segments."""
    transformed_dict = {}

    for key, value in d.items():
        if "_" in key:
            key.replace("_", sep)
        parts = key.split(sep)
        if len(parts) >2 :  # Only modify keys with exactly three segments
            new_key = f"{parts[0]}{sep}{parts[2]}"
        else:
            new_key = key  # Keep unchanged if not three segments
        transformed_dict[new_key] = value
    return transformed_dict

def postprocess_dict(d, support= None, prediction_mode = False):
    "alter the prediction mode to change the values for postprocessing and not post processing"

    if 'profile-name' in d:
        if isinstance(d['profile-name'], list):
            d['profile-name'] = ' '.join(d['profile-name']).lower().strip()
        else:
            d['profile-name'] = d['profile-name'].lower().strip()
        d['profile-name'] = d['profile-name'].lower().strip()
    if 'hotel-type' in d:
        if d.get('hotel-type', None) is not None and 'guest' in d.get('hotel-type', ' '):
            d['hotel-type'] = 'guesthouse'
    # removing keys with 'reqt' in them
    d = {k: v for k, v in d.items() if 'reqt' not in k}

    if not prediction_mode:
        return d
    
    keys_to_replace_with_ontology = ['train-departure', 'restaurant-food',  'train-departure', 'taxi-departure' ,'taxi-destination', 'hotel-name', 'attraction-type', 'restaurant-name', 'attraction-name', 'hospital-department']
    for  key in d.keys(): #  
        if key in keys_to_replace_with_ontology:
            if d[key] in ontology[key]:
                pass
            else:
                if 'train-destination' in key:
                    print('')
                try:
                    best_match, score = process.extractOne(d[key], ontology[key])
                    # print(f"Replaced '{d[key]}' with '{best_match}'  Actual : {support[key]}(Score: {score})")
                    d[key] = best_match
                except:
                    pass

    return d



    
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

import re

def parse_simple_xml_recursive(text: str) -> dict:
    """
    Recursively parse simple XML-like tags into a nested dictionary.
    """
    def convert_value(val: str):
        lower = val.lower()
        if lower in {"", "none", "null"}:
            return None
        if lower == "true":
            return True
        if lower == "false":
            return False
        return val

    def has_inner_tags(value: str) -> bool:
        return bool(re.search(r"<(\w+)>\s*.*?\s*</\1>", value, re.DOTALL))

    matches = re.findall(r"<(\w+)>\s*(.*?)\s*</\1>", text, re.DOTALL | re.IGNORECASE)
    result = {}

    for tag, raw in matches:
        content = raw.strip()
        if has_inner_tags(content):
            result[tag] = parse_simple_xml_recursive(content)
        else:
            result[tag] = convert_value(content)

    return result

def main(PATH):
    with open(PATH) as f:
        data = f.readlines()
    orig_data = [json.loads(line) for line in data]


    counter = 0
    scores = []
    error_vals = 0
    frequent_mistake_keys = {}
    frequent_mistake_examples = {}
    exact_match_count = 0
    bad_predictions_format = 0 


    for num, row in enumerate(orig_data):

        label = row['dialog_state']
        predict = row['llama_response']

        # label = extract_json_from_tag('DST', label)
        predict = parse_simple_xml_recursive(predict)
        if 'state' in predict.keys():
            predict = predict['state']

        if predict == {}:
            bad_predictions_format += 1
            continue
        if predict is None:
            continue

        # remove the keys and values that are empty in label and predictions
        label = {k: v for k, v in label.items() if v != {}}
        predict = {k: v for k, v in predict.items() if v != {}}

        # if you want to remove matching keys
        # temp_counter = 0
        # for key in label.keys():
        #     if label[key] == predict.get(key, {'a':0}):
        #         temp_counter += 1
        # # calculate the key level match
        # key_level_match = temp_counter / len(label.keys())


        # R_label = row['R_label']
        # R_predict = row['R_predict']

        R_label = flatten_dict(label)
        R_predict = flatten_dict(predict)

        R_label = transform_keys(R_label)
        R_predict = transform_keys(R_predict)

        R_label = postprocess_dict(R_label)

        R_predict = postprocess_dict(R_predict, R_label, prediction_mode = False)

        R_label_keys = list(R_label.keys())
        R_predict_keys = list(R_predict.keys())

        if R_label == R_predict:
            exact_match_count += 1

        temp_key = list(set(R_label_keys).intersection(R_predict_keys))

        counter = 0
        for key in temp_key:
            if R_label[key] == R_predict.get(key, {'a':0}):
                counter += 1
            else: 
                "Logging mismatched keys for error analysis here" 
                frequent_mistake_keys[key] = frequent_mistake_keys.get(key, 0) + 1
                
                if key not in frequent_mistake_examples:
                    frequent_mistake_examples[key] = []

                frequent_mistake_examples[key].append([{"label": R_label[key], "predict": R_predict[key], "num": num}])
            
        try:
            score = counter / len(temp_key)
            scores.append(score)
        except: 
            error_vals += 1
            pass

    sorted_mistakes = sorted(frequent_mistake_keys.items(), key=lambda x: x[1], reverse=True)
    print("Frequent Mistake Keys:")


    for key, value in sorted_mistakes:
        print(f"Key: {key}, Count: {value}")

    print('SCORE',np.mean(scores))
    print('Exact matches', exact_match_count)
    print('Bad predictions', bad_predictions_format)
    print('Total turns', len(orig_data))
    print('Error vals', error_vals)

    # with open(f'/home/srijithr/course_hw/Aura/dst/evals/{PATH.split("/")[-1].split(".")[0]}_NO_POSTPROCESSING.txt', 'w') as f:
    #     f.write(f"Frequent Mistake Keys:\n")
    #     for key, value in sorted_mistakes:
    #         f.write(f"Key: {key}, Count: {value}\n")
    #     f.write(f'Number of scored samples: {len(scores)}\n')
    #     f.write(f'SCORE: {np.mean(scores)}\n')
    #     f.write(f'Exact matches: {exact_match_count}\n')
    #     f.write(f'Bad predictions format from DST: {bad_predictions_format}\n')
    #     f.write(f'Total turns: {len(orig_data)}\n')
    #     f.write(f'Error vals: {error_vals}\n')





if __name__ == "__main__":
    main('data_for_llm/test_data_with_llama_70B_response.jsonl')
    # root_dir=  '/home/srijithr/course_hw/Aura/dst/outputs'
    # files = os.listdir(root_dir)
    # files = [file for file in files if file.endswith('.jsonl')]
    # files = [file for file in files if 'lora' in file]
    # for file in files:
    #     print(f'Processing {os.path.join(root_dir,file)}')
    #     main(os.path.join(root_dir,file))



    # """	•	restaurant-food: 55 → 43 (-12)
    #     •	train-departure: 51 → 32 (-19)
    #     •	taxi-departure: 42 → 32 (-10)
    #     •	taxi-destination: 41 → 27 (-14)
    #     •	hotel-name: 31 → 11 (-20)
    #     •	restaurant-name: 29 → 12 (-17)
    #     •	attraction-type: 30 → 25 (-5)
    #     •	attraction-name: 22 → 8 (-14)
    #     •	hospital-department: 4 → (Removed)"""