#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
 
# Execute program located in $HOME
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load cuDNN/8.6.0.163-CUDA-11.8.0
module load FFmpeg/4.4.2-GCCcore-11.3.0 
python -u foraging_world_v2.py > output_file.out