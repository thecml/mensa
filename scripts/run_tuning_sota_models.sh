#!/bin/bash

base_path=$(dirname "$0")            # relative
base_path=$(cd "$base_path" && pwd)  # absolutized and normalized
if [[ -z "$base_path" ]] ; then
  exit 1  # fail
fi
echo "$base_path"

dataset_names=('seer_se' 'mimic_me' 'rotterdam_me' 'proact_me' 'ebmt_me')

for dataset_name in "${dataset_names[@]}"; do
  echo "Running tuning scripts for dataset_name=$dataset_name"

  python3 "$base_path/../src/tuning/tune_coxnet_model.py" --dataset_name "$dataset_name"
  python3 "$base_path/../src/tuning/tune_weibull_aft_model.py" --dataset_name "$dataset_name"

  python3 "$base_path/../src/tuning/tune_coxboost_model.py" --dataset_name "$dataset_name"
  python3 "$base_path/../src/tuning/tune_coxph_model.py" --dataset_name "$dataset_name"
  python3 "$base_path/../src/tuning/tune_deephit_model.py" --dataset_name "$dataset_name"
  python3 "$base_path/../src/tuning/tune_deepsurv_model.py" --dataset_name "$dataset_name"
  python3 "$base_path/../src/tuning/tune_dsm_model.py" --dataset_name "$dataset_name"

  if [[ "$dataset_name" != "seer_se" ]]; then
    python3 "$base_path/../src/tuning/tune_hierarch_model.py" --dataset_name "$dataset_name"
  else
    python3 "$base_path/../src/tuning/tune_hierarch_model.py" --dataset_name "seer_cr"
  fi

  python3 "$base_path/../src/tuning/tune_mtlr_model.py" --dataset_name "$dataset_name"
  python3 "$base_path/../src/tuning/tune_rsf_model.py" --dataset_name "$dataset_name"

done
