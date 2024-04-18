#!/bin/bash
base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

models=("direct" "hierarch")
datasets=("als")
echo "=============================================================================================="
echo "Starting datasets tuning"
echo "=============================================================================================="
for model in ${models[@]}; do
  for dataset in ${datasets[@]}; do
      echo "Starting dataset run for <$model> on <$dataset>"
      python $base_path/../src/tuning/tune_single_models.py --dataset $dataset --model $model
      echo "Tuning <$model> <$dataset> done"
      echo -e "\n\n\n\n\n"
      echo "=============================================================================================="
      echo -e "\n\n\n\n\n"
  done
done
echo "Finished executing datasets"