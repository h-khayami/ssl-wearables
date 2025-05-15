#!/bin/bash

K_SHOT_VALUES=(5 20 50)
FLIP_PATHS=("/home/hossein/ssl-wearables/model_check_point/2025-05-02_09:40:41tmp.pt")
SPLIT=("held_one_subject_out_k_shot")

for flip_path in "${FLIP_PATHS[@]}"
do
  for split in "${SPLIT[@]}"
  do
    for k in "${K_SHOT_VALUES[@]}"
    do
        echo "Running with K_shot=$k and flip_net_path=$flip_path and split_method=$split"
        python downstream_task_evaluation.py k_shot=$k split_method=$split evaluation.flip_net_path="$flip_path" #> "data/logs/${split}_K${k}_flip_$(basename $flip_path).log" 2>&1
    done
  done
done