#!/bin/bash
#SBATCH --job-name=v22_Ultimate
#SBATCH --output=/data/EunJi/h22000561_psa/logs/train_v22_%j.out
#SBATCH --error=/data/EunJi/h22000561_psa/logs/train_v22_%j.err
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1

export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1

cd /data3/home/h22000561/psa_grading/

/home/sonic/anaconda3/bin/python train_v22_ultimate.py
