#!/bin/bash
TASK_NAME="mbpp"
CATEGORIES=("subtask" "domain")
for CATEGORY in ${CATEGORIES[*]}; do
    python generate_categories.py \
        --category ${CATEGORY} \
        --delete_old_responses \
        --few_shot \
        --few_shot_k 5 \
        --task_name $TASK_NAME \
        --output_dir data/${TASK_NAME}/generated_categories \
        --train_file data/${TASK_NAME}/original_mbpp_test.json \
        --max_requests_per_minute 600 \
        --max_tokens_per_minute 160000
done