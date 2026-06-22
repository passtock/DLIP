#!/bin/bash
#SBATCH --job-name=PSA_v3_Train        # 작업 이름
#SBATCH --output=/data/EunJi/h22000561_psa/logs/v3_train_%j.out  # 정상 출력 로그 저장 경로 (%j는 작업번호)
#SBATCH --error=/data/EunJi/h22000561_psa/logs/v3_train_%j.err   # 에러 로그 저장 경로
#SBATCH --time=72:00:00                # 최대 허용 시간 (넉넉하게 3일)
#SBATCH --gres=gpu:1                   # GPU 1대 사용 (A30)
#SBATCH --cpus-per-task=8              # 파이썬 코드의 num_workers=8과 맞춤

# 가상환경(h22000561-psa) 강제 활성화 및 라이브러리 경로 셋팅
source /home/sonic/anaconda3/etc/profile.d/conda.sh
conda activate h22000561-psa
export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:$LD_LIBRARY_PATH

echo "======================================================="
echo "🚀 sbatch 백그라운드 학습 시작!"
echo "======================================================="

# 파이썬 학습 코드 실행
python v_ultimate_pro_3.py
