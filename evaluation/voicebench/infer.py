import argparse
from datasets import load_dataset
from agent.controller.controller import Controller
from agent.controller.modes import Mode
import json
import os
from tqdm import tqdm
import hashlib
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Run VoiceBench evaluation')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--subset', type=str, default='openbookqa', help='VoiceBench subset to evaluate on')
    return parser.parse_args()

"""
LLM_OPTIONS taken from environment variables

"""

def run_inference(dataset, controller, output_path):
    completed_list =[]
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            completed_list = [json.loads(line)['id'] for line in f]

    for i, input in tqdm(enumerate(dataset)):
        # Create a unique ID by hashing the prompt
        sample_id = hashlib.md5(input['prompt'].encode()).hexdigest()

        if sample_id in completed_list:
            print(f"Skipping {sample_id} because it already exists")
            continue
        
        audio, sr = input['audio']['array'], input['audio']['sampling_rate']
        output, transcript = controller.qa_eval({'instruction':(audio, sr)})
    
        result = {
            'id': sample_id,
            'question': input['prompt'],
            'reference_answer': input['reference'],
            'model_answer': output[-1]['action']['payload'] if len(output) > 0 else None,
            'asr_transcript': transcript,
            'trajectory': output
        }

        with open(output_path, 'a') as f:
            f.write(json.dumps(result) + '\n')
        controller.reset()
    print(f"\nResults saved to {output_path}")

def main():
    args = parse_args()
    exp_name = args.exp_name if args.exp_name else f"{args.subset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = f"evaluation/evaluation_outputs/voicebench/{args.subset}/{exp_name}.jsonl"
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    dataset = load_dataset("hlt-lab/voicebench", args.subset, split="test")
    controller = Controller(operation_mode=Mode.QA_EVAL, io_mode=Mode.SPEECH_2_TEXT_CASCADED, max_iterations=3)
    
    run_inference(dataset, controller, output_path)

if __name__ == "__main__":
    main()
    # controller = Controller(operation_mode=Mode.QA_EVAL, io_mode=Mode.TEXT_2_TEXT_CASCADED, max_iterations=3)
    # output, transcript = controller.qa_eval({'instruction':'Who is the current president of South Korea?'})
    # print(output)
    # print(transcript)