
#!/bin/bash

#SBATCH --job-name=v22_Seq

#SBATCH --output=/data/EunJi/h22000561_psa/logs/seq_grid_%j.out

#SBATCH --error=/data/EunJi/h22000561_psa/logs/seq_grid_%j.err

#SBATCH --time=120:00:00

#SBATCH --gres=gpu:1



# 1. 환경 세팅

export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:$LD_LIBRARY_PATH

source /home/sonic/anaconda3/etc/profile.d/conda.sh

conda activate h22000561-psa



cd /data3/home/h22000561/psa_grading/



# 2. 🎯 여기서 원하는 조합을 마음대로 설정하세요!

BACKBONE="b4"

LEARNING_RATES=("1e-4" "3e-5")

GAMMAS=("2.0" "3.0")



# 3. 1개의 GPU 위에서 순서대로 하나씩 실행 (Sequential Loop)

for LR in "${LEARNING_RATES[@]}"; do

    for GM in "${GAMMAS[@]}"; do

        

        # 저장될 폴더 및 실험 이름

        EXP_NAME="seq_${BACKBONE}_lr${LR}_gm${GM}"

        

        echo "=================================================="

        echo "🚀 [순차 학습 시작] 조합: $EXP_NAME"

        echo "=================================================="

        

        # 파이썬 마스터 스크립트 실행

        python train_eval_v22_master.py --backbone $BACKBONE --lr $LR --gamma $GM --exp_name $EXP_NAME

        

        echo "✅ [순차 학습 완료] 성적표 저장 완료: $EXP_NAME"

        echo -e "\n"

        

    done

done



echo "🎉 모든 순차적 그리드 탐색이 완료되었습니다!"

