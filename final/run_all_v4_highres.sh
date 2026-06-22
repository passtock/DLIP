#!/bin/bash
#SBATCH --job-name=Train_v4_HighRes
#SBATCH --output=/data/EunJi/h22000561_psa/logs/train_v4_%j.out
#SBATCH --error=/data/EunJi/h22000561_psa/logs/train_v4_%j.err
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1

export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1

# 무조건 작업 폴더로 먼저 이동!
cd /data3/home/h22000561/psa_grading/

VERSIONS=("v17_2" "v18_2" "v19_2" "v20_2")

for VER in "${VERSIONS[@]}"; do
    echo "======================================================="
    echo "🚀 [고해상도 모드] Starting training for $VER ..."
    echo "======================================================="
    
    # 🎯 무조건 절대경로 파이썬 사용! (에러 방지)
    /home/sonic/anaconda3/bin/python universal_v4_trainer.py $VER
done

echo "🎉 모든 모델의 고해상도 학습이 완료되었습니다!"
