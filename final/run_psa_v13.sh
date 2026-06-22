#!/bin/bash
#SBATCH --job-name=psa_v13_adv
#SBATCH --output=/data/EunJi/h22000561_psa/out.psa_v13.%j.txt
#SBATCH --error=/data/EunJi/h22000561_psa/err.psa_v13.%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=normal

# 쉘 로그인 환경 강제 로드 및 가상환경 활성화
source /home/sonic/anaconda3/etc/profile.d/conda.sh
conda activate h22000561-psa

# 파이썬 고도화 파일 백그라운드 구동
python3 /home/h22000561/psa_grading/psa_v13_advanced.py
