#!/bin/bash

base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

results_path=$base_path/../results/trajectory_loss.csv
if [ -f "$results_path" ]; then
  rm $results_path
fi

seeds=(0 1 2 3 4)
use_trajectory=("True" "False")
dataset_names=('rotterdam_me' 'ebmt_me')

for seed in "${seeds[@]}"; do
  for ut in "${use_trajectory[@]}"; do
    for dataset_name in "${dataset_names[@]}"; do
        echo "Running with seed=$seed, use_trajectory=$ut, dataset_name=$dataset_name"
        if [ "$ut" = "True" ]; then
          python3 $base_path/../src/ablation/train_mensa_trajectory_loss.py --seed "$seed" --use_trajectory "$ut" --dataset_name "$dataset_name"
        else
          python3 $base_path/../src/ablation/train_mensa_trajectory_loss.py --seed "$seed" --dataset_name "$dataset_name"
        fi
    done
done