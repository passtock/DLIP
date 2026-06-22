#!/bin/bash
#SBATCH --job-name=v22_END
#SBATCH --output=/data/EunJi/h22000561_psa/logs/train_endgame_%j.out
#SBATCH --error=/data/EunJi/h22000561_psa/logs/train_endgame_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1

# 에러 방지용 리눅스 환경 변수 세팅
export LC_ALL=C
export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:$LD_LIBRARY_PATH
source /home/sonic/anaconda3/etc/profile.d/conda.sh
conda activate h22000561-psa

cd /data3/home/h22000561/psa_grading/

# 파이썬 실행
python train_eval_binary_endgame.py
