#!/bin/sh 
#SBATCH --gres=gpu:A6000:1
#SBATCH --partition=general
#SBATCH --mem=32GB
#SBATCH --time 1-23:55:00 
#SBATCH --job-name=speech_asr-owsm_v3.1_ebf_small
#SBATCH --error=/home/gganeshl/Aura/accent_adaptive_asr/logs/error-owsm_v3.1_ebf_small.err
#SBATCH --output=/home/gganeshl/Aura/accent_adaptive_asr/logs/output-owsm_v3.1_ebf_small.out
#SBATCH --mail-type=END
#SBATCH --mail-user=gganeshl@andrew.cmu.edu

conda init
conda activate speech

python /home/gganeshl/Aura/accent_adaptive_asr/asr_ft.py