import torch
import datasets
import espnetez as ez # ESPnet wrapper that simplifies integration. If you get an error when executing this cell, click Runtime -> Restart Session, and rerun from the beginning
import numpy as np
import librosa
from espnet2.bin.s2t_inference import Speech2Text # Core ESPnet module for pre-trained models
# Datasets library
from datasets import load_dataset, Audio
import os

print("Installation success!")

# Load and prepare dataset
print("Loading DTU54DL/common-accent dataset...")
dataset = load_dataset("DTU54DL/common-accent")
print(f"Dataset info: {dataset}")
dataset = dataset.rename_column("sentence", "transcription")
dataset = dataset.remove_columns(["accent"])
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))


DIR = f"/data/user_data/gganeshl/owsm_v3.1_ebf_small/exp"
os.makedirs(DIR, exist_ok=True)
EXP_DIR = f"/data/user_data/gganeshl/owsm_v3.1_ebf_small/exp/finetune"
os.makedirs(EXP_DIR, exist_ok=True)
STATS_DIR = f"/data/user_data/gganeshl/owsm_v3.1_ebf_small/exp/stats_finetune"
os.makedirs(STATS_DIR, exist_ok=True)

# Split dataset into train, validation, and test
train_dataset = dataset["train"]
test_dataset = dataset["test"]

train_dataset = dataset["train"].select(range(1500))
valid_dataset = dataset["train"].select(range(2000, 2000 + 100))
test_dataset = test_dataset.select(range(200))

print(f"Train size: {(train_dataset)}")
print(f"Validation size: {(valid_dataset)}")
print(f"Test size: {(test_dataset)}")

# FINETUNE_MODEL="espnet/owsm_v3.1_ebf_base"
FINETUNE_MODEL = "espnet/owsm_v3.1_ebf_small"
# FINETUNE_MODEL = "espnet/owsm_v3.2"
owsm_language="eng" # language code in ISO3

pretrained_model = Speech2Text.from_pretrained(
    FINETUNE_MODEL,
    lang_sym=f"<{owsm_language}>",
    beam_size=1,
    device='cuda'
)
torch.save(pretrained_model.s2t_model.state_dict(), '/data/user_data/gganeshl/owsm_v3.1_ebf_small/exp/original.pth')
pretrain_config = vars(pretrained_model.s2t_train_args)
tokenizer = pretrained_model.tokenizer
converter = pretrained_model.converter

'''
pretrained_model -> the pre-trained model we downloaded earlier
tokenizer -> Tokenizes raw text into subwords
converter -> Converts subwords into integer IDs for model input
'''

def tokenize(text):
    return np.array(converter.tokens2ids(tokenizer.text2tokens(text)))
data_info = {
    "speech": lambda d: d['audio']['array'].astype(np.float32), # 1-D raw waveform
    "text": lambda d: tokenize(f"<{owsm_language}><asr><notimestamps> {d['transcription']}"), 
    "text_prev": lambda d: tokenize("<na>"),
    "text_ctc": lambda d: tokenize(d['transcription']),
}

test_data_info = {
    "speech": lambda d: d['audio']['array'].astype(np.float32),
    "text": lambda d: tokenize(f"<{owsm_language}><asr><notimestamps> {d['transcription']}"),
    "text_prev": lambda d: tokenize("<na>"),
    "text_ctc": lambda d: tokenize(d['transcription']),
    "text_raw": lambda d: d['transcription'],
}
train_dataset = ez.dataset.ESPnetEZDataset(train_dataset, data_info=data_info)
valid_dataset = ez.dataset.ESPnetEZDataset(valid_dataset, data_info=data_info)
test_dataset = ez.dataset.ESPnetEZDataset(test_dataset, data_info=test_data_info)

# define model loading function
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model_fn(args):
  model = pretrained_model.s2t_model
  model.train()
  print(f'Trainable parameters: {count_parameters(model)}')
  return model

finetune_config = ez.config.update_finetune_config(
	's2t',
	pretrain_config,
	f"/home/gganeshl/Aura/accent_adaptive_asr/config/finetune.yaml"
)

# You can edit your config by changing the finetune.yaml file directly (but make sure you rerun this cell again!)
# You can also change it programatically like this
finetune_config['num_iters_per_epoch'] = 500

trainer = ez.Trainer(
    task='s2t',
    train_config=finetune_config,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    build_model_fn=build_model_fn, # provide the pre-trained model
    data_info=data_info,
    output_dir=EXP_DIR,
    stats_dir=STATS_DIR,
    ngpu=1
)

train_dataset.data_info

trainer.collect_stats() # collect audio/text length information to construct batches

trainer.train() # every 100 steps takes ~1 min