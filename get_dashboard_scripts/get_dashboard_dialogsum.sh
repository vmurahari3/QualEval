#!/bin/bash
TASK_NAME="knkarthick_dialogsum"
FEW_SHOT_K=2

TEMPERATURE=0.9
MODEL_NAMES=("text-curie-001" "text-davinci-002" "text-davinci-003")
for MODEL_NAME in ${MODEL_NAMES[*]}; do
    GENERATION_DIR="${MODEL_NAME}_topp_1.0_temperature_${TEMPERATURE}_few_shotk_${FEW_SHOT_K}"
    python get_dashboard.py \
    --task_name $TASK_NAME \
    --input_dir_generation_scores data/${TASK_NAME}/model_generations/${GENERATION_DIR}/scores \
    --input_dir_gt_scores data/${TASK_NAME}/GT_scores/ \
    --generation_file data/${TASK_NAME}/model_generations/${GENERATION_DIR}/generations.jsonl \
    --categories "subtask,domain" \
    --proficiency_metric "rougeL" \
    # --pretty_plot
done
