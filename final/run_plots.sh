#!/bin/bash
source /home/sonic/anaconda3/etc/profile.d/conda.sh
conda activate h22000561-psa
export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:$LD_LIBRARY_PATH

VERSIONS=("v17_3_yolo" "v18_3_yolo" "v19_3_yolo" "v20_3_yolo")

for VER in "${VERSIONS[@]}"; do
    python generate_plots.py $VER
done
