#!/bin/bash

# 환경 세팅 폴더 생성
mkdir -p /data/EunJi/h22000561_psa/logs
mkdir -p /data/EunJi/h22000561_psa/grid_scripts

# 테스트할 하이퍼파라미터 조합
BACKBONES=("b2" "b4")
LEARNING_RATES=("1e-4" "3e-5")
GAMMAS=("2.0" "3.0")

for BB in "${BACKBONES[@]}"; do
    for LR in "${LEARNING_RATES[@]}"; do
        for GM in "${GAMMAS[@]}"; do
            
            # 실험 이름 생성 (예: grid_b4_lr3e-5_gm2.0)
            EXP_NAME="grid_${BB}_lr${LR}_gm${GM}"
            SCRIPT_NAME="/data/EunJi/h22000561_psa/grid_scripts/submit_${EXP_NAME}.sh"
            
            # 각각의 실험을 위한 고유 sbatch 스크립트 동적 생성
            cat << SCRIPT_EOF > $SCRIPT_NAME
#!/bin/bash
#SBATCH --job-name=${BB}_${GM}
#SBATCH --output=/data/EunJi/h22000561_psa/logs/${EXP_NAME}_%j.out
#SBATCH --error=/data/EunJi/h22000561_psa/logs/${EXP_NAME}_%j.err
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1

export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:\$LD_LIBRARY_PATH
source /home/sonic/anaconda3/etc/profile.d/conda.sh
conda activate h22000561-psa

cd /data3/home/h22000561/psa_grading/

# 마스터 파이썬 파일 실행 (인자 전달)
python train_eval_v22_master.py --backbone $BB --lr $LR --gamma $GM --exp_name $EXP_NAME
SCRIPT_EOF

            # 실행 권한 부여 및 큐 제출
            chmod +x $SCRIPT_NAME
            sbatch $SCRIPT_NAME
            
            echo "✅ 작업 제출 완료: $EXP_NAME"
        done
    done
done
