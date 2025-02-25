#!/bin/bash

base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

results_path=$base_path/../results/complexcity_competing_risks_dgp.csv
if [ -f "$results_path" ]; then
  rm $results_path
fi

python3 $base_path/../src/experiments/calculate_complexcity_competing_risks_dgp.py