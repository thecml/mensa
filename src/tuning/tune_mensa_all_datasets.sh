#!/bin/bash

base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

dataset_names=('seer_se' 'mimic_se' 'rotterdam_me' 'proact_me' 'ebmt_me')

for dataset_name in "${dataset_names[@]}"; do
  echo "Running dataset_name=$dataset_name"
  python3 $base_path/../src/tuning/tune_mensa_model.py --dataset_name "$dataset_name"
done