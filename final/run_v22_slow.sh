#!/bin/bash
#SBATCH --job-name=v22_Slow
#SBATCH --output=/data/EunJi/h22000561_psa/logs/train_v22_slow_%j.out
#SBATCH --error=/data/EunJi/h22000561_psa/logs/train_v22_slow_%j.err
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1

export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1

cd /data3/home/h22000561/psa_grading/

# 🎯 자동으로 기존 v22 스크립트를 읽어 학습률을 3e-5로 하향 조정
python -c "
with open('train_v22_ultimate.py', 'r') as f:
    code = f.read()
code = code.replace('lr=1e-4', 'lr=3e-5')
code = code.replace('v22_ultimate', 'v22_slow_learn')
with open('train_v22_slow_run.py', 'w') as f:
    f.write(code)
"

# 훈련 시작!
/home/sonic/anaconda3/bin/python train_v22_slow_run.py
