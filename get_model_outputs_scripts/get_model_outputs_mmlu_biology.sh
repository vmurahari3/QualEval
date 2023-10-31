#!/bin/bash
TASK_NAME="mmlu_biology"
FEW_SHOT_K=2
MODEL_NAMES=("text-curie-001" "text-davinci-002" "text-davinci-003")
TEMPERATURE=0.9
for MODEL_NAME in ${MODEL_NAMES[*]}; do
    python model_generations_driver.py \
        --delete_old_responses \
        --few_shot \
        --few_shot_k 2  \
        --task_name $TASK_NAME \
        --output_dir data/${TASK_NAME}/model_generations/ \
        --train_file data/mmlu_biology/original_mmlu_biology_test.json \
        --model $MODEL_NAME \
        --max_requests_per_minute 600 \
        --max_tokens_per_minute 200000 \
        --temperature ${TEMPERATURE}
done