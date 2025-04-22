# Install required libraries
import os
import shutil
import json
import torch
from huggingface_hub import HfApi, HfFolder
from espnet2.bin.s2t_inference import Speech2Text

# Set your HF username and model name
HF_USERNAME = "reecursion"  
MODEL_NAME = "accent-adaptive-owsm_v3.1_ebf_base"  
REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"

# Create directory structure for the model
model_dir = "./hf_model"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(f"{model_dir}/espnet_model", exist_ok=True)

# Load and save the fine-tuned model
# First, load your fine-tuned model
finetuned_model_path = "/home/gganeshl/speech/exp/finetune/valid.acc.best.pth"
finetuned_weights = torch.load(finetuned_model_path)

# Save the model weights
torch.save(finetuned_weights, f"{model_dir}/espnet_model/model.pth")

# Save configuration files
# Copy the config.yaml from the original model
original_model = "espnet/owsm_v3.1_ebf_base"
pretrained_model = Speech2Text.from_pretrained(
    original_model,
    lang_sym=f"<eng>",
    beam_size=1,
    device='cpu'  # Use CPU to avoid CUDA issues during setup
)

# Create model config
model_config = {
    "base_model": "espnet/owsm_v3.1_ebf_base",
    "language": "eng",
    "task": "asr",
    "description": "Fine-tuned OWSM model on common-accent dataset",
    "framework": "espnet"
}

with open(f"{model_dir}/config.json", "w") as f:
    json.dump(model_config, f, indent=4)

# Copy any necessary tokenizer files
tokenizer_dir = os.path.dirname(pretrained_model.tokenizer.__dict__.get('_model_file', ''))
if tokenizer_dir:
    os.makedirs(f"{model_dir}/tokenizer", exist_ok=True)
    for file in os.listdir(tokenizer_dir):
        shutil.copy(os.path.join(tokenizer_dir, file), f"{model_dir}/tokenizer/{file}")

# Create a readme file
readme = f"""# Common Accent ASR Model

This is a fine-tuned ASR model based on [espnet/owsm_v3.1_ebf_base](https://huggingface.co/espnet/owsm_v3.1_ebf_base) trained on the [DTU54DL/common-accent](https://huggingface.co/datasets/DTU54DL/common-accent) dataset.

## Model details
- Base model: espnet/owsm_v3.1_ebf_base
- Language: English
- Task: Automatic Speech Recognition

## Usage

```python
import torch
import numpy as np
from espnet2.bin.s2t_inference import Speech2Text

# Load the model
model = Speech2Text.from_pretrained(
    "{REPO_ID}",
    lang_sym="<eng>",
    beam_size=1,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Example inference
waveform = ...  # Load your audio as numpy array
transcription = model(waveform)
print(transcription[0][0])  # Print the transcription"""


# Write the readme to a file
with open(f"{model_dir}/README.md", "w") as f:
    f.write(readme)

# Upload to Hugging Face
api = HfApi()

# Login to Hugging Face - you'll need to set your token first
# You can use HF_TOKEN environment variable or login interactively
# token = "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"
# if token:
#     api.set_access_token(token)
# else:
#     # You'll need to run huggingface-cli login once before this
#     token = HfFolder.get_token()
#     if not token:
#         print("Please login using `huggingface-cli login` or set HF_TOKEN environment variable")
#         exit(1)

# Create the repository (if it doesn't exist)
try:
    api.create_repo(repo_id=REPO_ID, exist_ok=True)
    print(f"Repository {REPO_ID} is ready")
except Exception as e:
    print(f"Error creating repository: {e}")
    exit(1)

# Upload files to Hugging Face
api.upload_folder(
    folder_path=model_dir,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload fine-tuned OWSM model",
)

print(f"Model uploaded successfully! Access it at: https://huggingface.co/{REPO_ID}")