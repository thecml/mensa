#!/bin/bash
base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi

echo "$base_path"
results_path=$base_path/../results/single_event_dgp.csv
if [ -f "$results_path" ]; then
  rm $results_path
fi

seeds=(0 1 2 3 4)
k_taus=(0.0 0.25 0.50 0.75)
copula_names=("clayton" "frank")
linear=("True" "False")

for copula_name in "${copula_names[@]}"; do
    for lin in "${linear[@]}"; do
        for k_tau in "${k_taus[@]}"; do
            for seed in "${seeds[@]}"; do
                echo "Running with seed=$seed, k_tau=$k_tau, copula_name=$copula_name, linear=$lin"
                if [ "$lin" = "True" ]; then
                    python3 $base_path/../src/experiments/train_models_single_event_dgp.py --seed "$seed" --k_tau "$k_tau" --copula_name "$copula_name" --linear
                else
                    python3 $base_path/../src/experiments/train_models_single_event_dgp.py --seed "$seed" --k_tau "$k_tau" --copula_name "$copula_name"
                fi
            done
        done
    done
done
