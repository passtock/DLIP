#!/bin/bash
#SBATCH --job-name=Train_v19_v20
#SBATCH --output=/data/EunJi/h22000561_psa/logs/train_19_20_%j.out
#SBATCH --error=/data/EunJi/h22000561_psa/logs/train_19_20_%j.err
#SBATCH --time=72:00:00  # 2개이므로 3일로 단축
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

source /home/sonic/anaconda3/etc/profile.d/conda.sh
conda activate h22000561-psa
export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:$LD_LIBRARY_PATH

# 딱 두 가지만 실행합니다!
VERSIONS=("v19_2" "v20_2")

for VER in "${VERSIONS[@]}"; do
    echo "======================================================="
    echo ">>> Starting training for $VER on YOLO 10K dataset..."
    echo "======================================================="
    python universal_v3_trainer.py $VER
done

echo "🎉 v19와 v20의 새로운 v3 학습 및 평가가 완료되었습니다!"
