import argparse
import json
import random
import torch
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_jsonl(file_path, num_samples=None):
    """Load JSONL file and return specified number of random conversations.
    If num_samples is None, returns all conversations."""
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    if num_samples is None:
        return data
    return random.sample(data, min(num_samples, len(data)))

def get_user_messages(conversations):
    """Extract the user messages from conversations as a list."""
    messages = []
    for msg in conversations:
        if msg.get('role') == 'user':
            messages.append(msg)
            break
    return messages

def main():
    parser = argparse.ArgumentParser(description='Generate text using a finetuned model')
    parser.add_argument('--model_name_or_path', type=str, required=True,
                      help='Base model name or path')
    parser.add_argument('--checkpoint_dir', type=str,
                      help='Directory containing the finetuned checkpoint. If not provided, model_name_or_path will be used.')
    parser.add_argument('--input_file', type=str, 
                      default='data/finetune/filtered_data/val.jsonl',
                      help='Path to input JSONL file containing conversations')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Path to save the generated responses')
    parser.add_argument('--num_samples', type=int, 
                      help='Number of input conversations to process. If not provided, all samples will be used.')
    args = parser.parse_args()

    # Load model and tokenizer
    model_path = args.checkpoint_dir if args.checkpoint_dir else args.model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Get conversations from input data
    samples = load_jsonl(args.input_file, args.num_samples)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Open output file for writing
    with open(args.output_file, 'w') as f:
        for idx, sample in enumerate(tqdm(samples, desc="Processing samples"), 1):
            print(f"\n\nProcessing sample {idx}/{len(samples)}")
            
            # Extract the user messages from conversations
            messages = get_user_messages(sample['messages'])
            if not messages:
                print("No user message found in this sample, skipping...")
                continue
                
            print("\nInput message:")
            print(messages[0]['content'])
            
            # Format the messages using the chat template
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            
            # Generate response
            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True).to(model.device)
            
            print("\nGenerating response...")
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=10000,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("\nGenerated response:")
            print(response)
            
            # Save to output file
            output_data = {
                'input': messages[0],
                'response': response,
                'sample_id': idx,
                'generation_metrics': {
                    'temperature': 0.7,
                    'top_p': 0.95,
                    'top_k': 50,
                    'max_new_tokens': 10000
                },
                'model_info': {
                    'checkpoint_dir': args.checkpoint_dir if args.checkpoint_dir else args.model_name_or_path,
                    'model_name_or_path': args.model_name_or_path
                }
            }
            f.write(json.dumps(output_data) + '\n')

if __name__ == "__main__":
    main() 