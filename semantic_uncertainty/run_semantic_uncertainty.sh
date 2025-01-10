#!/bin/bash
#SBATCH --job-name=semantic_uncertainty
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --time=02:00:00
#SBATCH --partition=whartonstat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --constraint=node9

source activate semantic_uncertainty
python semantic_uncertainty/generate_answers.py --model_name=Llama-2-7b-chat --dataset=trivia_qa
