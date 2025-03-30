#!/bin/sh
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=256Gb
#SBATCH --cpus-per-task=40
#SBATCH -t 2-00:00:00              # time limit:  add - for days (D-HH:MM)
#SBATCH --job-name=generate_aura_llmfac_qwen2_5_1_5b_instruct-lora-epochs4-lr1e-5-10_5kcontext-grad_accum4_perdev4
#SBATCH --error=/home/lmaben/cw/sentientia/Aura/logs/generate/job_outputs/%x__%j.err
#SBATCH --output=/home/lmaben/cw/sentientia/Aura/logs/generate/job_outputs/%x__%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lmaben@andrew.cmu.edu    

source /data/tir/projects/tir7/user_data/lmaben/miniconda3/etc/profile.d/conda.sh
conda activate code-edit-bench
cd /home/lmaben/cw/sentientia/Aura

export HF_DATASETS_CACHE=/data/tir/projects/tir7/user_data/lmaben/hf_cache_dir
export HF_HOME=/data/tir/projects/tir7/user_data/lmaben/hf_home_dir

export NCCL_P2P_DISABLE=1

python dst/generate/generate.py --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
--checkpoint_dir /data/tir/projects/tir7/user_data/lmaben/llm-weights/outputs/aura/aura_llmfac_qwen2_5_1_5b_instruct-lora-epochs4-lr1e-5-10_5kcontext-grad_accum4_perdev4/checkpoint-272 \
--input_file data/spokenwoz_dst/test.jsonl \
--output_file /data/tir/projects/tir7/user_data/lmaben/llm-weights/outputs/aura/generate/aura_llmfac_qwen2_5_1_5b_instruct-lora-epochs4-lr1e-5-10_5kcontext-grad_accum4_perdev4_ckpt_272/test_output.jsonl

