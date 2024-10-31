#!/bin/bash
#SBATCH -J submit         # 任务名称
#SBATCH -N 1             # 计算节点个数
#SBATCH -p bme_gpu       # 队列名
#SBATCH -n 8             # CPU核心数，不建议更改此处，避免读写降速
#SBATCH --gres=gpu:1     # 申请GPU个数
#SBATCH -t 96:00:00      # 任务运行时间上限，max=120:00:00
#SBATCH -o submit.out   # 任务输出log
LD_LIBRARY_PATH=/public/software/anaconda/anaconda3/2023.09/lib:$LD_LIBRARY_PATH 

echo ${SLURM_JOB_NODELIST} # 打印任务所出的计算节点

module load cuda/7/11.8
source ~/.bashrc
source /public/software/anaconda/anaconda3/2023.09/etc/profile.d/conda.sh

nvidia-smi                 # 查看GPU占用情况，申请到这张卡后是独享，卡上应该无其他任务
echo start on $(date)      # 开始时间
conda activate lt-dinov2     # 激活你的conda环境

export PYTHONPATH=$PYTHONPATH:$(pwd)    

which python

# python /home_data/home/v-luotao/projects/pretrain/dinov2/extra_generator.py

python dinov2/run/train/train.py \
    --nodes 1 --gpus 1 --partition bme_gpu \
    --config-file dinov2/configs/train/vitb14_panoramic.yaml \
    --output-dir ./output/vitb14_panoramic \
    train.dataset_path=ImageNet:split=TRAIN:root=dinov2/data/ImageNet-1k-style:extra=dinov2/data/extra

module unload cuda/7/11.8

conda deactivate