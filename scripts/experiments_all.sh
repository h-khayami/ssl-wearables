#!/bin/bash

K_SHOT_VALUES=(5 20 50)
LR=0.0001
run_eval () {
    for k in "${K_SHOT_VALUES[@]}"
    do
        LOG_FILE="data/logs/${NAME// /_}K${k}_LR${LR}.log"

        if [[ -f "$LOG_FILE" ]] && grep -q "Classification report saved to" "$LOG_FILE"; then
            echo "✅ Skipping K_shot=$k (already completed: $LOG_FILE)"
            continue
        fi

        echo "🚀 Running with K_shot=$k, flip_net_path=$FLIP_PATH, split_method=$SPLIT"
        python downstream_task_evaluation.py \
            k_shot=$k \
            split_method=$SPLIT \
            evaluation.learning_rate=$LR \
            evaluation.load_weights=$LOAD_WEIGHTS \
            evaluation.flip_net_path="$FLIP_PATH" \
            evaluation.load_personalized_weights=$LOAD_PERSONALIZED \
            > "$LOG_FILE" 2>&1
    done
}


# --- First config ---
NAME="MLP_OPT-Personalized"
FLIP_PATH="/data/ssl_wearable/model/2025-05-02_09:40:41tmp.pt"
SPLIT="k_shot_fixed"
LOAD_WEIGHTS="True"
LOAD_PERSONALIZED="True"
run_eval

# --- Second config ---
NAME="MLP_Broader-OPT"
FLIP_PATH="/data/ssl_wearable/model/2025-04-14_22:45:26tmp.pt"
SPLIT="held_one_subject_out_k_shot"
LOAD_WEIGHTS="True"
LOAD_PERSONALIZED="False"
run_eval

# --- Third config ---
NAME="MLP_Broader-Personalized"
FLIP_PATH="/data/ssl_wearable/model/2025-04-14_22:45:26tmp.pt"
SPLIT="k_shot_fixed"
LOAD_WEIGHTS="True"
LOAD_PERSONALIZED="False"
run_eval

# --- Fourth config ---
NAME="MLP_Broader-OPT-Personalized"
FLIP_PATH="/data/ssl_wearable/model/2025-04-15_13:15:18tmp.pt"
SPLIT="k_shot_fixed"
LOAD_WEIGHTS="True"
LOAD_PERSONALIZED="True"
run_eval

# --- Fifth config (no pretrained weights) ---
NAME="MLP_Personalized"
FLIP_PATH=""  # not using pretrained model
SPLIT="k_shot_fixed"
LOAD_WEIGHTS="False"
LOAD_PERSONALIZED="False"
run_eval