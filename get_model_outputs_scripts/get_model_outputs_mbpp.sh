#!/bin/bash
TASK_NAME="mbpp"
FEW_SHOT_K=2
MODEL_NAMES=("text-davinci-002" "text-davinci-003")
TEMPERATURE=0.9
for MODEL_NAME in ${MODEL_NAMES[*]}; do
    python model_generations_driver.py \
        --delete_old_responses \
        --few_shot \
        --few_shot_k $FEW_SHOT_K \
        --task_name $TASK_NAME \
        --output_dir data/${TASK_NAME}/model_generations/ \
        --train_file data/mbpp/original_mbpp_test.json \
        --model $MODEL_NAME \
        --max_requests_per_minute 600 \
        --max_tokens_per_minute 200000 \
        --temperature ${TEMPERATURE}
done