#!/bin/bash
#SBATCH -J psa_train
#SBATCH -o /home/h22000561/psa_grading/logs/out_%j.txt
#SBATCH -e /home/h22000561/psa_grading/logs/err_%j.txt
#SBATCH -p normal
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1

echo "### START: $(date) ###"
source /home/h22000561/miniconda3/etc/profile.d/conda.sh
conda activate h22000561-psa
cd ~/psa_grading
python psa_v5.py
echo "### END: $(date) ###"
