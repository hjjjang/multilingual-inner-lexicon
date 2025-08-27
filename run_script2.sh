#!/bin/bash

#SBATCH --job-name=script2  # Job name
#SBATCH --ntasks-per-node=1  # Number of tasks per node
#SBATCH --cpus-per-task=10  # Number of CPU cores per task
#SBATCH --mem=120G  # Total memory per node
#SBATCH --mail-type=BEGIN,END,FAIL  # Notifications for job begin, end, and failure
#SBATCH --gres=gpu:1  # Request 1 GPU
#SBATCH --partition=gpu-vram-48gb  # Use the 48GB GPU partition
#SBATCH --time=12:00:00  # Job time limit 

# Load conda environment (assuming conda is installed and available in your system)
source /work/hyujang/miniconda3/etc/profile.d/conda.sh  # Change this path to where your conda is installed
conda activate thesis  # Replace 'your_conda_env' with your environment name

# Run your Python script
python /home/hyujang/multilingual-inner-lexicon/RQ1/WordIdentity/patchscope_test2.py