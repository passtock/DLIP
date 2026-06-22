#!/bin/bash
#SBATCH --job-name=Train_v20
#SBATCH --output=/data/EunJi/h22000561_psa/logs/train_v20_alone_%j.out
#SBATCH --error=/data/EunJi/h22000561_psa/logs/train_v20_alone_%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1

export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:$LD_LIBRARY_PATH

echo "======================================================="
echo "🚀 멈췄던 끝판왕 v20 단독 학습 시작!"
echo "======================================================="

cd /data3/home/h22000561/psa_grading/

# 방금 찾은 진짜 파이썬 절대 경로 적용!
/home/sonic/anaconda3/bin/python universal_v3_trainer.py v20_2
