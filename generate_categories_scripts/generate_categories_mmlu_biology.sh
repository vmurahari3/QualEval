#!/bin/bash
TASK_NAME="mmlu_biology"
CATEGORIES=("subtask" "domain")
for CATEGORY in ${CATEGORIES[*]}; do
    python generate_categories.py \
        --category ${CATEGORY} \
        --delete_old_responses \
        --few_shot \
        --few_shot_k 5 \
        --task_name $TASK_NAME \
        --output_dir data/$TASK_NAME/generated_categories \
        --train_file data/${TASK_NAME}/original_mmlu_biology_test.json \
        --max_requests_per_minute 600 \
        --max_tokens_per_minute 160000 \
        --num_categories_pruning_factor 4
done