#!/bin/bash
#SBATCH --job-name=ultimate_pro_A30
#SBATCH --output=/data3/home/h22000561/psa_grading/test_result.out
#SBATCH --error=/data3/home/h22000561/psa_grading/test_result.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

export TMPDIR="/data3/home/h22000561/psa_grading/.tmp"
export TORCH_HOME="/data3/home/h22000561/psa_grading/.cache/torch"
export TIMM_HOME="/data3/home/h22000561/psa_grading/.cache/timm"

echo ">>> Start Ultimate PRO Job (A30 Optimized) <<<"
/home/sonic/anaconda3/bin/python -u /data3/home/h22000561/psa_grading/v_ultimate_pro.py
echo ">>> Job Finished <<<"
