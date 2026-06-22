#!/bin/bash
#SBATCH --job-name=v22_SeqB4
#SBATCH --output=/data/EunJi/h22000561_psa/logs/seq_grid_b4_%j.out
#SBATCH --error=/data/EunJi/h22000561_psa/logs/seq_grid_b4_%j.err
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1

# 1. 안전한 환경 세팅
export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:$LD_LIBRARY_PATH
source /home/sonic/anaconda3/etc/profile.d/conda.sh
conda activate h22000561-psa

cd /data3/home/h22000561/psa_grading/

# 2. 🎯 백본을 B4로 변경 및 안전한 하이퍼파라미터 세팅
BACKBONE="b4"
LEARNING_RATES=("3e-5") # B4의 체급을 고려하여 가장 안전한 저속 학습률 고정
GAMMAS=("2.0" "3.0")    # 감마(어려운 문제 집중도)만 2가지로 교차 테스트

# 3. 단 1개의 GPU에서 순서대로 학습 (순차 탐색)
for LR in "${LEARNING_RATES[@]}"; do
    for GM in "${GAMMAS[@]}"; do
        
        # 실험 이름 생성 (예: seq_b4_lr3e-5_gm2.0)
        EXP_NAME="seq_${BACKBONE}_lr${LR}_gm${GM}"
        
        echo "=================================================="
        echo "🚀 [B4 순차 학습 시작] 조합: $EXP_NAME"
        echo "=================================================="
        
        # 파이썬 마스터 스크립트 실행 (미리 고쳐둔 에러 없는 버전!)
        python train_eval_v22_master.py --backbone $BACKBONE --lr $LR --gamma $GM --exp_name $EXP_NAME
        
        echo "✅ [B4 순차 학습 완료] 성적표 저장 완료: $EXP_NAME"
        echo -e "\n"
        
    done
done

echo "🎉 B4 순차적 그리드 탐색이 모두 완료되었습니다!"
