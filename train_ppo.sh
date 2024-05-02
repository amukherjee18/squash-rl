#!/bin/bash -l
#SBATCH -J train_ppo
#SBATCH --gres=gpu:1
#SBATCH -t 7:00:00
#SBATCH --mem=120GB
#SBATCH --output=slurm/%x-%j.out

conda activate lightlm-cuda12.1

python train_ppo.py