#!/bin/bash
#SBATCH --job-name=PSA_Fixed_Train
#SBATCH --output=/data/EunJi/h22000561_psa/logs/v3_fixed_train_%j.out
#SBATCH --error=/data/EunJi/h22000561_psa/logs/v3_fixed_train_%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

source /home/sonic/anaconda3/etc/profile.d/conda.sh
conda activate h22000561-psa
export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:$LD_LIBRARY_PATH

echo "======================================================="
echo "🚀 버그 픽스 및 학습 보고서 기능 탑재 sbatch 시작!"
echo "======================================================="

python v_ultimate_pro_3_fixed.py
