#!/bin/bash

base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

results_path=$base_path/../results/real_me.csv
if [ -f "$results_path" ]; then
  rm $results_path
fi

seeds=(0 1 2 3 4)
dataset_names=('mimic_me' 'als_me')

for seed in "${seeds[@]}"; do
    for dataset_name in "${dataset_names[@]}"; do
        echo "Running with seed=$seed, dataset_name=$dataset_name"
        python3 $base_path/../src/experiments/run_real_multi_event.py --seed "$seed" --dataset_name "$dataset_name"
    done
done