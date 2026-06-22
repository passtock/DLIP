#!/bin/bash
#SBATCH --job-name=Train_v21
#SBATCH --output=/data/EunJi/h22000561_psa/logs/train_v21_%j.out
#SBATCH --error=/data/EunJi/h22000561_psa/logs/train_v21_%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

source /home/sonic/anaconda3/etc/profile.d/conda.sh
conda activate h22000561-psa
export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:$LD_LIBRARY_PATH

# 위에서 만든 파이썬 파일 실행!
python train_v21_tuned.py
