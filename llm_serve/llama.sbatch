#!/bin/sh
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=500Gb
#SBATCH --cpus-per-task=40
#SBATCH -t 2-00:00:00              # time limit:  add - for days (D-HH:MM)
#SBATCH --job-name=serve_llama_70B
#SBATCH --error=/home/lmaben/code-edit-bench/logs/llm_serve/drafts/job_outputs/%x__%j.err
#SBATCH --output=/home/lmaben/code-edit-bench/logs/llm_serve/drafts/job_outputs/%x__%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lmaben@andrew.cmu.edu    
source /data/tir/projects/tir7/user_data/lmaben/miniconda3/etc/profile.d/conda.sh
conda activate code-edit-bench
cd /home/lmaben/code-edit-bench

export HF_TOKEN=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

export HF_DATASETS_CACHE=/data/tir/projects/tir7/user_data/lmaben/hf_cache_dir
export VLLM_CACHE_DIR=/data/tir/projects/tir7/user_data/lmaben/vllm_cache_dir
export HF_HOME=/data/tir/projects/tir7/user_data/lmaben/hf_home_dir
export NCCL_P2P_DISABLE=1

vllm serve "meta-llama/Llama-3.3-70B-Instruct" --tensor-parallel-size 8 --host 0.0.0.0 --port 8000