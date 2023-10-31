from utils.templates import  PROFICIENCY_METRICS
from evaluate import load
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
code_eval = load("code_eval")
rouge = load("rouge")

def get_mbpp_proficiency_score(generated_text, sample):
    proficiency_metrics = PROFICIENCY_METRICS['mbpp']
    output_metrics = {}
    for metric in proficiency_metrics:
        if metric == 'pass@1':
            test_cases = sample['test_list']
            candidates = [[generated_text]] * len(test_cases)
            output, _ = code_eval.compute(references=test_cases, predictions=candidates, k=[1])
            output_metrics[metric] = output['pass@1']
        else:
            raise ValueError("Metric not supported")
    return output_metrics

def get_dialogsum_proficiency_score(generated_text, sample):
    proficiency_metrics = PROFICIENCY_METRICS['knkarthick_dialogsum']
    output_metrics = {}
    predictions = [generated_text]
    references = [sample['summary']]
    rouge_scores = rouge.compute(references=references, predictions=predictions)
    for metric in proficiency_metrics:
        output_metrics[metric] = rouge_scores[metric] 
    return output_metrics

def get_mmlu_biology_proficiency_score(generated_text, sample):
    # calculate accuracy
    proficiency_metrics = PROFICIENCY_METRICS['mmlu_biology']
    output_metrics = {}
    for metric in proficiency_metrics:
        if metric == 'accuracy':
            prediction = generated_text.strip()
            reference = sample['answer'].strip()
            if prediction == reference:
                output_metrics[metric] = 1
            else:
                output_metrics[metric] = 0
        else:
            raise ValueError("Metric not supported")
    return output_metrics


def get_medmcqa_proficiency_score(generated_text, sample):
    # calculate accuracy
    proficiency_metrics = PROFICIENCY_METRICS['medmcqa']
    output_metrics = {}
    for metric in proficiency_metrics:
        if metric == 'accuracy':
            prediction = generated_text.strip()
            reference = sample['answer'].strip()
            if prediction == reference:
                output_metrics[metric] = 1
            else:
                output_metrics[metric] = 0
        else:
            raise ValueError("Metric not supported")
    return output_metrics

def get_proficiency_score(generated_text, sample, task_name):
    assert task_name in PROFICIENCY_METRICS.keys(), "Task name not supported"
    if task_name == 'mbpp':
        return get_mbpp_proficiency_score(generated_text, sample)
    elif task_name == 'knkarthick_dialogsum':
        return get_dialogsum_proficiency_score(generated_text, sample)
    elif task_name == 'mmlu_biology':
        return get_mmlu_biology_proficiency_score(generated_text, sample)
    elif task_name == 'medmcqa':
        return get_medmcqa_proficiency_score(generated_text, sample)
    else:
        raise ValueError("Task name not supported")
     