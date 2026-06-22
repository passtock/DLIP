#!/bin/bash
#SBATCH --job-name=v22_B4
#SBATCH --output=/data/EunJi/h22000561_psa/logs/train_v22_b4_%j.out
#SBATCH --error=/data/EunJi/h22000561_psa/logs/train_v22_b4_%j.err
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1

# 1. 안전한 디렉토리 및 환경 설정
mkdir -p /data/EunJi/h22000561_psa/logs
export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:$LD_LIBRARY_PATH

# 2. 콘다 환경 강제 활성화 (중요!)
source /home/sonic/anaconda3/etc/profile.d/conda.sh
conda activate h22000561-psa

cd /data3/home/h22000561/psa_grading/

# 3. B4 파이썬 코드 생성 및 실행
python -c "
with open('train_v22_ultimate.py', 'r') as f:
    code = f.read()
code = code.replace('efficientnet_b2', 'efficientnet_b4')
code = code.replace('EfficientNet_B2_Weights', 'EfficientNet_B4_Weights')
code = code.replace('v22_ultimate', 'v22_b4_upgrade')
with open('train_v22_b4_run.py', 'w') as f:
    f.write(code)
"

python train_v22_b4_run.py
