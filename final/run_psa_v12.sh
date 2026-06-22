#!/bin/bash
#SBATCH -J psa_v12_advanced
#SBATCH -o /data/EunJi/h22000561_psa/out.psa_v12.%j.txt
#SBATCH -e /data/EunJi/h22000561_psa/err.psa_v12.%j.txt
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1

source /home/h22000561/miniconda3/etc/profile.d/conda.sh
conda activate h22000561-psa

cd /home/h22000561/psa_grading
python3 -u psa_v12_advanced.py
