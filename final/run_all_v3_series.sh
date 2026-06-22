#!/bin/bash
#SBATCH --job-name=Train_v17_to_v20
#SBATCH --output=/data/EunJi/h22000561_psa/logs/train_all_v3_%j.out
#SBATCH --error=/data/EunJi/h22000561_psa/logs/train_all_v3_%j.err
#SBATCH --time=120:00:00  # 4개를 돌리므로 넉넉하게 5일 세팅
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

source /home/sonic/anaconda3/etc/profile.d/conda.sh
conda activate h22000561-psa
export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:$LD_LIBRARY_PATH

# 🚀 v17_2, v18_2, v19_2, v20_2의 아키텍처를 순서대로 불러와서 10K 데이터로 재학습!
VERSIONS=("v17_2" "v18_2" "v19_2" "v20_2")

for VER in "${VERSIONS[@]}"; do
    echo "======================================================="
    echo ">>> Starting training for $VER on YOLO 10K dataset..."
    echo "======================================================="
    python universal_v3_trainer.py $VER
done

echo "🎉 모든 모델(v17~v20)의 새로운 v3 학습 및 평가가 완료되었습니다!"
