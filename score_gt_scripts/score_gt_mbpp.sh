#!/bin/bash
TASK_NAME="mbpp"
GENERATION_DIR=text-davinci-002_topp_1.0_temperature_0.9_few_shotk_2
python score_categories.py \
--generation_field "code" \
--input_field "input_to_model" \
--generation_file data/${TASK_NAME}/model_generations/${GENERATION_DIR}/generations.jsonl \
--subtask_file data/${TASK_NAME}/generated_categories/api_generated_dataset_subtask.jsonl \
--domain_file data/${TASK_NAME}/generated_categories/api_generated_dataset_domain.jsonl \
--task_name $TASK_NAME \
--input_dir data/${TASK_NAME} \
--output_dir data/${TASK_NAME}/GT_scores \
--delete_old_responses \
--max_requests_per_minute 500 \
--max_tokens_per_minute 140000 