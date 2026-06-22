#!/bin/sh
#SBATCH -J psa_v6_training
#SBATCH -o out.psa_v6.%j.txt
#SBATCH -e err.psa_v6.%j.txt
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1

source /home/h22000561/miniconda3/etc/profile.d/conda.sh
conda activate h22000561-psa

cd /home/h22000561/psa_grading
python3 psa_v6.py
