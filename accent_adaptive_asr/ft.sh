#!/bin/sh 
#SBATCH --gres=gpu:A6000:1
#SBATCH --partition=general
#SBATCH --mem=32GB
#SBATCH --time 1-23:55:00 
#SBATCH --job-name=speech_asr
#SBATCH --error=/home/gganeshl/speech/logs/error.err
#SBATCH --output=/home/gganeshl/speech/logs/output.out
#SBATCH --mail-type=END
#SBATCH --mail-user=gganeshl@andrew.cmu.edu

conda init
conda activate speech

python /home/gganeshl/speech/asr_ft.py