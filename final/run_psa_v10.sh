#!/bin/sh
#SBATCH -J psa_v10_heavy
#SBATCH -o out.psa_v10.%j.txt
#SBATCH -e err.psa_v10.%j.txt
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1

source /home/h22000561/miniconda3/etc/profile.d/conda.sh
conda activate h22000561-psa

cd /home/h22000561/psa_grading
# FIXED: Added -u to force real-time unbuffered log flushing
python3 -u psa_v10_final.py
