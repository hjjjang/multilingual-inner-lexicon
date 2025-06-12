#!/bin/bash

#SBATCH --job-name=myjob-batch  # Job name
#SBATCH --ntasks-per-node=1  # Number of tasks per node
#SBATCH --cpus-per-task=10  # Number of CPU cores per task
#SBATCH --mem=70G  # Total memory per node
#SBATCH --mail-type=BEGIN,END,FAIL  # Notifications for job begin, end, and failure
#SBATCH --gres=gpu:1  # Request 1 GPU
#SBATCH --partition=gpu-vram-48gb  # Use the 48GB GPU partition
#SBATCH --time=4:00:00  # Job time limit (2 hours)

# Load conda environment (assuming conda is installed and available in your system)
source /work/hyujang/miniconda3/etc/profile.d/conda.sh  # Change this path to where your conda is installed
conda activate thesis  # Replace 'your_conda_env' with your environment name

# Run your Python script
python RQ1/classification3.py  # Replace with the full path to your Python file
