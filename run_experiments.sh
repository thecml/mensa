#!/bin/bash

# 定义参数值
thetas=(0.5 1.333 3 8)
initial_thetas=(1 2)

# 遍历参数值并运行 Python 脚本 None, CyclicLR, ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR, 
for theta in "${thetas[@]}"; do
    for initial_theta in "${initial_thetas[@]}"; do
        echo "Running with theta=$theta and initial_theta=$initial_theta"
        python3 test_script_weijie.py --theta_dgp "$theta" --initial_theta "$initial_theta" --LR_optimal CosineAnnealingLR
    done
done