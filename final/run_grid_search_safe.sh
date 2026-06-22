#!/bin/bash
mkdir -p /data/EunJi/h22000561_psa/logs
mkdir -p /data/EunJi/h22000561_psa/grid_scripts

BACKBONES=("b2")
LEARNING_RATES=("3e-5") # 🎯 터지는 1e-4 삭제! 안전한 저속 학습만 진행
GAMMAS=("2.0" "3.0")

for BB in "${BACKBONES[@]}"; do
    for LR in "${LEARNING_RATES[@]}"; do
        for GM in "${GAMMAS[@]}"; do
            EXP_NAME="grid_${BB}_lr${LR}_gm${GM}"
            SCRIPT_NAME="/data/EunJi/h22000561_psa/grid_scripts/submit_${EXP_NAME}.sh"
            
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
python train_eval_v22_master.py --backbone $BB --lr $LR --gamma $GM --exp_name $EXP_NAME
SCRIPT_EOF

            chmod +x $SCRIPT_NAME
            sbatch $SCRIPT_NAME
            echo "✅ 안전 작업 제출 완료: $EXP_NAME"
        done
    done
done
