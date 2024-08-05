#!/bin/bash

base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

results_path=$base_path/../results/synthetic_se.csv
if [ -f "$results_path" ]; then
  rm $results_path
fi

copula_names=("clayton" "frank" "gumbel")
linear=("True" "False")
k_taus=(0.0 0.2 0.4 0.6 0.8)
seeds=(0 1 2 3 4)

for copula_name in "${copula_names[@]}"; do
    for lin in "${linear[@]}"; do
        for k_tau in "${k_taus[@]}"; do
            for seed in "${seeds[@]}"; do
                echo "Running with seed=$seed, k_tau=$k_tau, copula_name=$copula_name, linear=$lin"
                python3 $base_path/../src/experiments/run_dgp_single_event.py --seed "$seed" --k_tau "$k_tau" --copula_name "$copula_name" --linear "$lin"
            done
        done
    done
done