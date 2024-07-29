#!/bin/bash

file_path="../results/synthetic_se.csv"

if [ -f "$file_path" ]; then
  rm "$file_path"
else
  echo "File $file_path does not exist."
fi

copula_names=("clayton" "frank" "gumbel")
linearities=("True" "False")
k_taus=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
seeds=(0 1 2 3 4)

for copula_name in "${copula_names[@]}"; do
    for linearity in "${linearities[@]}"; do
        for k_tau in "${k_taus[@]}"; do
            for seed in "${seeds[@]}"; do
                echo "Running with seed=$seed, k_tau=$k_tau, copula_name=$copula_name, linearity=$linearity"
                python3 run_dgp_single_event.py --seed "$seed" --k_tau "$k_tau" --copula_name "$copula_name" --linear "$linearity"
            done
        done
    done
done