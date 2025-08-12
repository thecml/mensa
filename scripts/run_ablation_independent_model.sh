#!/bin/bash

base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

results_path=$base_path/../results/independent_model.csv
if [ -f "$results_path" ]; then
  rm $results_path
fi

seeds=($(seq 0 9))
independent_model=("True" "False")
dataset_names=('rotterdam_me' 'proact_me')

for seed in "${seeds[@]}"; do
    for im in "${independent_model[@]}"; do
        for dataset_name in "${dataset_names[@]}"; do
            echo "Running with seed=$seed, independent_model=$im, dataset_name=$dataset_name"
            if [ "$im" = "True" ]; then
                python3 $base_path/../src/ablation/train_mensa_independent_model.py --seed "$seed" --independent_model --dataset_name "$dataset_name"
            else
                python3 $base_path/../src/ablation/train_mensa_independent_model.py --seed "$seed" --dataset_name "$dataset_name"
            fi
        done
    done
done
