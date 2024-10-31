#!/bin/bash

# Parameters
#SBATCH --error=/home_data/home/v-luotao/projects/pretrain/dinov2/output/vitb14_panoramic/%j_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=dinov2:train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/home_data/home/v-luotao/projects/pretrain/dinov2/output/vitb14_panoramic/%j_0_log.out
#SBATCH --partition=bme_gpu
#SBATCH --signal=USR2@120
#SBATCH --time=2800
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home_data/home/v-luotao/projects/pretrain/dinov2/output/vitb14_panoramic/%j_%t_log.out --error /home_data/home/v-luotao/projects/pretrain/dinov2/output/vitb14_panoramic/%j_%t_log.err /public_bme/data/v-luotao/conda_envs/lt-dinov2/bin/python -u -m submitit.core._submit /home_data/home/v-luotao/projects/pretrain/dinov2/output/vitb14_panoramic
