#!/bin/sh
#SBATCH -J psa_grid_search
#SBATCH -o out.psa_grid.%j.txt
#SBATCH -e err.psa_grid.%j.txt
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1

source /home/h22000561/miniconda3/etc/profile.d/conda.sh
conda activate h22000561-psa

cd /home/h22000561/psa_grading

# 파라미터 조합 (Grid Search)
for LR in 1e-4 3e-4
do
  for BS in 16 24
  do
    echo "========================================="
    echo "Running with LR: $LR, Batch Size: $BS"
    echo "========================================="
    
    python3 psa_v9_tune.py --lr $LR --batch_size $BS --epochs 40 --margin 1.3
    
  done
done

echo "All grid search experiments completed!"
