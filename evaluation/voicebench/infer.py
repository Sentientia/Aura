import argparse
from datasets import load_dataset
from agent.controller.controller import Controller
from agent.controller.modes import Mode
import json
import os
from tqdm import tqdm
import hashlib
from datetime import datetime

MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
ASR_MODEL = os.getenv("ASR_MODEL", "Whisper v3 Large")


def parse_args():
    parser = argparse.ArgumentParser(description='Run VoiceBench evaluation')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--subset', type=str, default='alpacaeval_full', help='VoiceBench subset to evaluate on')
    return parser.parse_args()

"""
LLM_OPTIONS taken from environment variables

"""

def run_inference_qa(dataset, controller, output_path, additional_instruction: str=None):
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
        output, transcript = controller.qa_eval({'instruction':(audio, sr), 'additional_instruction':additional_instruction})
    
        result = {
            'id': sample_id,
            'question': input['prompt'],
            'reference_answer': input['reference'] if 'reference' in input else 'N/A',
            'model_answer': output[-1]['action']['payload'] if len(output) > 0 else None,
            'asr_transcript': transcript,
            'trajectory': output,
            'model': MODEL,
            'asr_model': ASR_MODEL
        }

        with open(output_path, 'a') as f:
            f.write(json.dumps(result) + '\n')
        controller.reset()
    print(f"\nResults saved to {output_path}")


INFERENCE_FN_MAP = {
    "openbookqa": run_inference_qa,
    "alpacaeval": run_inference_qa,
    "alpacaeval_full": run_inference_qa,
    "commoneval": run_inference_qa,
}

def main():
    args = parse_args()
    exp_name = args.exp_name if args.exp_name else f"{args.subset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = f"evaluation/evaluation_outputs/voicebench/{args.subset}/{exp_name}.jsonl"
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset = load_dataset("hlt-lab/voicebench", args.subset, split="test")

    if args.subset in  ["openbookqa", "alpacaeval", "alpacaeval_full","commoneval"]:
        controller = Controller(operation_mode=Mode.QA_EVAL, io_mode=Mode.SPEECH_2_TEXT_CASCADED, max_iterations=3)
        run_inference_fn = INFERENCE_FN_MAP[args.subset]
        if args.subset == "openbookqa":
            additional_instruction = "Always perform ATLEAST ONE web search before answering the question"
            run_inference_fn(dataset, controller, output_path, additional_instruction=additional_instruction)
        elif args.subset in ["alpacaeval", "alpacaeval_full","commoneval"]:
            # additional_instruction = '''You are a helpful assistant. You need to provide relevant, accurate  and to the point answer to the question.'''
#             additional_instruction = '''Answer the question in a way that is exceptionally relevant, accurate and to the point. 
# It should address the user's query in a highly effective manner, providing exactly the information needed.
# Always perform ATLEAST ONE web search before answering the question'''
            additional_instruction = '''Answer the question in a way that is exceptionally relevant, accurate and to the point. 
It should address the user's query in a highly effective manner, providing exactly the information needed.'''
            run_inference_fn(dataset, controller, output_path, additional_instruction=additional_instruction)
    else:
        raise ValueError(f"Invalid subset: {args.subset}")

if __name__ == "__main__":
    main()
    # controller = Controller(operation_mode=Mode.QA_EVAL, io_mode=Mode.TEXT_2_TEXT_CASCADED, max_iterations=3)
    # output, transcript = controller.qa_eval({'instruction':'Who is the current president of South Korea?'})
    # print(output)
    # print(transcript)