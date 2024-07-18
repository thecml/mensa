#!/bin/bash

initial_thetas=(2)
kendall_taus=(0.25 0.3)
survival_models=('Weibull_log_linear' 'Exp_linear' 'EXP_nonlinear' 'LogNormal_linear' 'LogNormal_nonlinear' 'Weibull_nonlinear', 'Weibull_linear')

for kendall_tau in "${kendall_taus[@]}"; do
    for initial_theta in "${initial_thetas[@]}"; do
        for survival_model in "${survival_models[@]}"; do
            echo "Running with kendall_tau=$kendall_tau and initial_theta=$initial_theta and survival_model=$survival_model"
            python3 train_model_general.py --model "$survival_model" --initial_theta "$initial_theta" --KENDALL_TAUS "$kendall_tau" --num_epoch 500
        done
    done
done