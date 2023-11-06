#!/bin/bash
#SBATCH --gpus 1
#SBATCH -p gpu_c128
#SBATCH -c 16
module load anaconda/2021.05 cuda/11.3 gcc/6.3
source activate torch
date
tar -xf ../clean.tar -C /dev/shm
date
python train_model.py --embed_dim 128