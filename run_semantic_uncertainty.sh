#!/bin/bash
#SBATCH --job-name=code
#SBATCH --time=00-14:00:00
#SBATCH --partition=whartonstat
#SBATCH --mem=16G
#SBATCH --gpus=2

conda activate semantic_uncertainty
python semantic_uncertainty/generate_answers.py --model_name=Llama-2-7b-chat --dataset=trivia_qa
