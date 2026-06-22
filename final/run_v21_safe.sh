#!/bin/bash
#SBATCH --job-name=Train_v21
#SBATCH --output=/data/EunJi/h22000561_psa/logs/train_v21_safe_%j.out
#SBATCH --error=/data/EunJi/h22000561_psa/logs/train_v21_safe_%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1

export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:$LD_LIBRARY_PATH

echo "======================================================="
echo "🚀 [안전 모드] v21 (EfficientNet + Transformer) 학습 시작"
echo "======================================================="

cd /data3/home/h22000561/psa_grading/

# 방금 찾은 진짜 파이썬 절대 경로 적용! (메모리 방지용 batch_size 8)
/home/sonic/anaconda3/bin/python train_v21_tuned.py --batch_size 8 --num_workers 4
