#!/bin/sh
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:4
#SBATCH --mem=256Gb
#SBATCH --cpus-per-task=40
#SBATCH -t 2-00:00:00              # time limit:  add - for days (D-HH:MM)
#SBATCH --job-name=aura_llmfac_qwen2_5_1_5b_instruct-lora-epochs4-lr1e-5-10_5kcontext-grad_accum4_perdev4
#SBATCH --error=/home/lmaben/cw/sentientia/Aura/logs/sft_llmfac/job_outputs/%x__%j.err
#SBATCH --output=/home/lmaben/cw/sentientia/Aura/logs/sft_llmfac/job_outputs/%x__%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lmaben@andrew.cmu.edu    

source /data/tir/projects/tir7/user_data/lmaben/miniconda3/etc/profile.d/conda.sh
conda activate code-edit-bench
cd /home/lmaben/cw/sentientia/Aura/LLaMA-Factory

export HF_DATASETS_CACHE=/data/tir/projects/tir7/user_data/lmaben/hf_cache_dir
export HF_HOME=/data/tir/projects/tir7/user_data/lmaben/hf_home_dir
export WANDB_API_KEY=e31b45068f2a99d234d3e0f6a5619c150beefd41
export NCCL_P2P_DISABLE=1

FORCE_TORCHRUN=1 llamafactory-cli train /home/lmaben/cw/sentientia/Aura/dst/finetune/configs/lora_finetune_qwen1_5b.yaml

## torchrun  --standalone --nnodes=1 --nproc-per-node=8  src/train.py /home/lmaben/code-edit-bench/finetune_llama_factory/configs/full_finetune.yaml