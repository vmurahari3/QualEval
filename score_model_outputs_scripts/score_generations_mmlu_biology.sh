#!/bin/bash
TASK_NAME="mmlu_biology"
FEW_SHOT_K=2
TEMPERATURE=0.9
MODEL_NAMES=("text-curie-001" "text-davinci-002" "text-davinci-003")

for MODEL_NAME in ${MODEL_NAMES[*]}; do
    GENERATION_DIR="${MODEL_NAME}_topp_1.0_temperature_${TEMPERATURE}_few_shotk_${FEW_SHOT_K}"
    python score_categories.py \
    --generation_field "generation" \
    --input_field "input_to_model" \
    --generation_file data/${TASK_NAME}/model_generations/${GENERATION_DIR}/generations.jsonl \
    --subtask_file data/${TASK_NAME}/generated_categories/api_generated_dataset_subtask.jsonl \
    --domain_file data/${TASK_NAME}/generated_categories/api_generated_dataset_domain.jsonl \
    --task_name ${TASK_NAME} \
    --input_dir data/${TASK_NAME} \
    --output_dir data/${TASK_NAME}/model_generations/${GENERATION_DIR}/scores \
    --delete_old_responses \
    --max_requests_per_minute 60 \
    --max_tokens_per_minute 140000
done