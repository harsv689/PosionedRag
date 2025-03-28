#!/bin/bash

# export TRANSFORMERS_CACHE="/scratch0/sghosal2/.cache/"
# export HF_HOME="/scratch0/sghosal2/.cache/huggingface/"

# Parameters
eval_model_code="contriever"
split="test"
query_results_dir="main"
model_name="llama7b"
use_truth="False"
top_k=5
gpu_id=0
attack_method="hotflip"
adv_per_query=5
score_function="dot"
repeat_times=10
M=10
seed=12
note=""

# Datasets to iterate over
# datasets=("nq" "hotpotqa" "msmarco")
datasets=("nq")
for eval_dataset in "${datasets[@]}"
do
    echo $eval_dataset
    log_dir="logs/${query_results_dir}_logs"
    mkdir -p "$log_dir"

    if [ "$use_truth" == "True" ]; then
        log_name="${eval_dataset}-${eval_model_code}-${model_name}-Truth--M${M}x${repeat_times}"
    else
        log_name="${eval_dataset}-${eval_model_code}-${model_name}-Top${top_k}--M${M}x${repeat_times}"
    fi

    if [ -n "$attack_method" ]; then
        log_name="${log_name}-adv-${attack_method}-${score_function}-${adv_per_query}-${top_k}"
    fi

    if [ -n "$note" ]; then
        log_name="$note"
    fi

    log_file="${log_dir}/${log_name}.txt"

    python main.py \
        --eval_model_code "$eval_model_code" \
        --eval_dataset "$eval_dataset" \
        --split "$split" \
        --query_results_dir "$query_results_dir" \
        --model_name "$model_name" \
        --top_k "$top_k" \
        --use_truth "$use_truth" \
        --gpu_id "$gpu_id" \
        --attack_method "$attack_method" \
        --adv_per_query "$adv_per_query" \
        --score_function "$score_function" \
        --repeat_times "$repeat_times" \
        --M "$M" \
        --seed "$seed" \
        # --name "$log_name" > "$log_file" 

done
