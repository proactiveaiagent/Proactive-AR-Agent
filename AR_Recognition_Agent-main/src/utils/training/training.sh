#!/bin/bash
#SBATCH -J finetune          # 作业名
#SBATCH -p gpu-share               # 分区
#SBATCH -c 20                       # CPU cores per task
#SBATCH --ntasks=1                 # 单任务
#SBATCH --gres=gpu:rtx6000:10      # 申请 8 张 RTX6000 GPU
#SBATCH --output=slurm-%j.log      # 输出文件
#SBATCH --error=slurm-%j.log       # 错误文件

# ======================
# 环境准备
# ======================
source /opt/ohpc/pub/apps/anaconda3-2024.02/etc/profile.d/conda.sh
# 激活环境
conda activate finetune

# 可选：确认环境 & Python
echo "Running on host: $(hostname)"
echo "Using python: $(which python)"
python --version

# ======================
# 运行程序
# ======================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash sft_7b.sh
