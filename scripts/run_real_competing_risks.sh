#!/bin/bash

file_path="../results/real_cr.csv"

if [ -f "$file_path" ]; then
  rm "$file_path"
else
  echo "File $file_path does not exist."
fi

seeds=(0 1 2 3 4)
dataset_names=('seer_cr' 'rotterdam_cr')

for seed in "${seeds[@]}"; do
    for dataset_name in "${dataset_names[@]}"; do
        echo "Running with seed=$seed, dataset_name=$dataset_name"
        python3 run_real_competing_risks.py --seed "$seed" --dataset_name "$dataset_name"
    done
    done
done