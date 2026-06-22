#!/bin/bash
#SBATCH --job-name=PSA_v4_Trans
#SBATCH --output=/data/EunJi/h22000561_psa/logs/v4_train_%j.out
#SBATCH --error=/data/EunJi/h22000561_psa/logs/v4_train_%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1

source /home/sonic/anaconda3/etc/profile.d/conda.sh
conda activate h22000561-psa
export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:$LD_LIBRARY_PATH

python v_ultimate_pro_4_transparent.py
