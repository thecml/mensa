#!/bin/bash

base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

results_path=$base_path/../results/shared_layer.csv
if [ -f "$results_path" ]; then
  rm $results_path
fi

seeds=($(seq 0 24))
use_shared=("True" "False")
dataset_names=('proact_me' 'rotterdam_me')

for seed in "${seeds[@]}"; do
    for us in "${use_shared[@]}"; do
        for dataset_name in "${dataset_names[@]}"; do
            echo "Running with seed=$seed, use_shared=$us, dataset_name=$dataset_name"
            if [ "$us" = "True" ]; then
                python3 $base_path/../src/ablation/train_mensa_shared_layer.py --seed "$seed" --use_shared --dataset_name "$dataset_name"
            else
                python3 $base_path/../src/ablation/train_mensa_shared_layer.py --seed "$seed" --dataset_name "$dataset_name"
            fi
        done
    done
done
