#!/bin/sh
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:8
#SBATCH --mem=256Gb
#SBATCH --cpus-per-task=40
#SBATCH -t 2-00:00:00              # time limit:  add - for days (D-HH:MM)
#SBATCH --job-name=qwen_2_5_1_5B_full_1e-6_24k_grad_accum32_llmfac
#SBATCH --error=/home/lmaben/code-edit-bench/logs/sft_llmfac/job_outputs/%x__%j.err
#SBATCH --output=/home/lmaben/code-edit-bench/logs/sft_llmfac/job_outputs/%x__%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lmaben@andrew.cmu.edu    

source /data/tir/projects/tir7/user_data/lmaben/miniconda3/etc/profile.d/conda.sh
conda activate code-edit-bench
cd /home/lmaben/LLaMA-Factory

export HF_DATASETS_CACHE=/data/tir/projects/tir7/user_data/lmaben/hf_cache_dir
export HF_HOME=/data/tir/projects/tir7/user_data/lmaben/hf_home_dir
export WANDB_API_KEY=XXXXXX
export NCCL_P2P_DISABLE=1

FORCE_TORCHRUN=1 llamafactory-cli train /home/lmaben/code-edit-bench/finetune_llama_factory/configs/full_finetune.yaml

## torchrun  --standalone --nnodes=1 --nproc-per-node=8  src/train.py /home/lmaben/code-edit-bench/finetune_llama_factory/configs/full_finetune.yaml