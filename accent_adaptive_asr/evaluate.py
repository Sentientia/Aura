import os
import torch
import torchaudio
import numpy as np
from espnet2.bin.s2t_inference import Speech2Text
from tqdm import tqdm
import jiwer
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset, Audio

# Initialize model
FINETUNE_MODEL = "espnet/owsm_v3.1_ebf_base"
owsm_language = "eng"  # language code in ISO3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Loading model from {FINETUNE_MODEL} on {device}")
pretrained_model = Speech2Text.from_pretrained(
    FINETUNE_MODEL,
    lang_sym=f"<{owsm_language}>",
    beam_size=1,
    device=device
)

# Save original model weights for comparison
original_path = 'original.pth'
if not os.path.exists(original_path):
    print(f"Saving original model weights to {original_path}")
    torch.save(pretrained_model.s2t_model.state_dict(), original_path)

# Get model components
pretrain_config = vars(pretrained_model.s2t_train_args)
tokenizer = pretrained_model.tokenizer
converter = pretrained_model.converter


def evaluate_checkpoint(model, checkpoint_path, test_data):
    """Evaluate a specific checkpoint on test data"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.s2t_model.load_state_dict(checkpoint)
    model.s2t_model.eval()
    
    references = []
    predictions = []
    sample_results = []
    
    # Regex pattern to remove tags
    import re
    tag_pattern = re.compile(r'<[^>]+>')
    
    for idx, sample in enumerate(tqdm(test_data)):
        try:
            # Extract audio data and process it
            speech = sample['audio']['array']
            reference = sample['transcription']
            
            # Run inference
            pred = model(speech)
            raw_prediction = pred[0][0]
            
            # Clean prediction by removing tags
            cleaned_prediction = re.sub(tag_pattern, '', raw_prediction).strip()
            
            references.append(reference)
            predictions.append(cleaned_prediction)
            
            # Store detailed results for the first few samples
            if idx < 5:
                sample_results.append({
                    'reference': reference,
                    'raw_prediction': raw_prediction,
                    'prediction': cleaned_prediction
                })
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            print(f"Sample keys: {sample.keys()}")
            continue
    
    # Calculate metrics if we have any valid results
    if len(references) > 0:
        wer = jiwer.wer(references, predictions)
        cer = jiwer.cer(references, predictions)
    else:
        print("No valid predictions were made!")
        wer = float('nan')
        cer = float('nan')
    
    return {
        'wer': wer,
        'cer': cer,
        'samples': sample_results
    }

def main():
    # Load and prepare dataset
    print("Loading DTU54DL/common-accent dataset...")
    dataset = load_dataset("DTU54DL/common-accent")
    print(f"Dataset info: {dataset}")
    
    # Rename and process columns
    dataset = dataset.rename_column("sentence", "transcription")
    dataset = dataset.remove_columns(["accent"])
    
    
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    # Use the test split for evaluation
    test_data = dataset["test"]
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Define checkpoints to evaluate
    checkpoints = [
        {"name": "Original", "path": "original.pth"},
        {"name": "Checkpoint", "path": "./exp/finetune/valid.acc.best.pth"}
    ]
    
    # Filter to only include checkpoints that exist
    checkpoints = [cp for cp in checkpoints if os.path.exists(cp["path"])]
    
    # Evaluate each checkpoint
    results = []
    for checkpoint in checkpoints:
        print(f"\nEvaluating {checkpoint['name']} checkpoint")
        result = evaluate_checkpoint(pretrained_model, checkpoint["path"], test_data)
        result["name"] = checkpoint["name"]
        results.append(result)
        
        # Print results
        print(f"Results for {checkpoint['name']}:")
        print(f"WER: {result['wer']:.4f}")
        print(f"CER: {result['cer']:.4f}")
        
        # Print example predictions
        print("\nExample predictions:")
        for idx, sample in enumerate(result['samples']):
            print(f"Sample {idx+1}:")
            print(f"REFERENCE: {sample['reference']}")
            print(f"PREDICTED: {sample['prediction']}")
            print("")
    
    # Create comparison table
    results_df = pd.DataFrame([
        {
            "Checkpoint": r["name"],
            "WER": r["wer"],
            "CER": r["cer"]
        } for r in results
    ])
    
    print("\n=== Results Summary ===")
    print(results_df)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # WER plot
    plt.subplot(1, 2, 1)
    checkpoint_names = [r["name"] for r in results]
    wer_values = [r["wer"] for r in results]
    plt.bar(checkpoint_names, wer_values)
    plt.title("Word Error Rate (WER)")
    plt.xlabel("Checkpoint")
    plt.ylabel("WER")
    plt.xticks(rotation=45)
    
    # CER plot
    plt.subplot(1, 2, 2)
    cer_values = [r["cer"] for r in results]
    plt.bar(checkpoint_names, cer_values)
    plt.title("Character Error Rate (CER)")
    plt.xlabel("Checkpoint")
    plt.ylabel("CER")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("asr_evaluation_results.png")
    print("Results plot saved to asr_evaluation_results.png")

if __name__ == "__main__":
    main()