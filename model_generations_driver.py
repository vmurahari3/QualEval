import os
import argparse
import random
import numpy as np
from utils.misc_utils import get_prompt, authenticate, seed_function
from datasets import load_dataset
import logging
import json
from api_request_parallel_processor import process_api_requests_from_file
import asyncio
from utils.args import add_args
from utils.proficiency_scores import get_proficiency_score

# Import global variables
from utils.templates import DEMONSTRATION_TEMPLATE, LABEL_KEY, TASK_INSTRUCTIONS


#  function to get the prompt by randomly sampling k samples from the train dataset
def get_generation_prompts(
    args,
    train_dataset,
    task_instruction,
    demonstration_template,
    incontext_dataset=None,
):
    all_prompts = []
    task_label_key = LABEL_KEY[args.task_name]
    task_label_key_formatted = "{" + task_label_key + "}"
    demonstration_template_without_label = demonstration_template.split(
        task_label_key_formatted
    )[0]
    for i in range(len(train_dataset)):
        cur_example = train_dataset[i]
        cur_example.pop(task_label_key)
        assert (
            task_label_key_formatted in demonstration_template
        ), "Task label key not found in demonstration template"
        cur_example_formatted = demonstration_template_without_label.format(
            **cur_example
        )
        if incontext_dataset is not None:
            assert len(incontext_dataset) == len(
                train_dataset
            ), "Incontext dataset must be the same size as the train dataset"
            demos = incontext_dataset[i]["demos"]
            demos = "\n".join(demos)
            prompt = task_instruction + "\n" + demos + "\n" + cur_example_formatted
            all_prompts.append(
                {
                    "prompt": prompt,
                    "sample": cur_example_formatted,
                    "demos": demos,
                    "task_instruction": task_instruction,
                    "id": i,
                }
            )
        else:
            if args.few_shot_k > 0:
                # randomly sample k samples from the train dataset for in-context examples
                incontext_indices = random.sample(
                    range(len(train_dataset)), args.few_shot_k
                )
                while i in incontext_indices:
                    incontext_indices = random.sample(
                        range(len(train_dataset)), args.few_shot_k
                    )
                incontext_train = train_dataset.select(
                    incontext_indices, args.few_shot_k
                )
                template_collated_demos, collated_demos = get_prompt(
                    args, incontext_train, task_instruction, demonstration_template
                )

            else:
                template_collated_demos = task_instruction
                collated_demos = ""

            # add the current example to the prompt
            template_collated_demos = (
                template_collated_demos + "\n" + cur_example_formatted
            )

            all_prompts.append(
                {
                    "prompt": template_collated_demos,
                    "sample": cur_example_formatted,
                    "demos": collated_demos,
                    "task_instruction": task_instruction,
                    "id": i,
                }
            )
    return all_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generations_file",
        type=str,
        default=None,
        help="If specified, will use generations from this file instead of generating new ones",
    )
    parser.add_argument(
        "--incontext_dataset_file",
        type=str,
        default=None,
        help="If specified, will use this dataset for incontext examples",
    )
    parser = add_args(parser)
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=args.logging_level)

    # Random seed
    seed_function(args)
    # Authenticate
    api_key = authenticate(args)

    if args.generations_file is not None:
        # overwrite the generations file and append the proficiency score to the generations from the file
        generations_file = args.generations_file
        # load the jsonl file
        with open(generations_file, "r") as f:
            generations = [json.loads(line) for line in f]
        # get the proficiency score for each generation
        # add the proficiency score to the generations
        generated_samples = []
        for generation in generations:
            proficiency_metrics = get_proficiency_score(
                generation["generation"], generation, args.task_name
            )
            for metric_name, metric_value in proficiency_metrics.items():
                generation[f"generation_{metric_name}"] = metric_value
            generated_samples.append(generation)
        # write the generations to the file
        with open(generations_file, "w") as f:
            for generation in generated_samples:
                f.write(json.dumps(generation) + "\n")
        return
    cur_run_dir = f"{args.model}_topp_{args.top_p}_temperature_{args.temperature}_few_shotk_{args.few_shot_k}"
    if args.incontext_sampling_strategy == "fixed" and args.train_file is None:
        raise ValueError(
            "incontext_sampling_strategy is fixed but train_file is not specified. Not clear which samples to use for in-context examples."
        )
    # load the dataset, load from local file if path is specified, otherwise download from the hub
    # post loading, massage the datasets to a custom format
    if args.train_file is not None:
        # check extension
        if args.train_file.endswith(".jsonl") or args.train_file.endswith(".json"):
            train_dataset = load_dataset("json", data_files=args.train_file)["train"]
        elif args.train_file.endswith(".tsv"):
            print(args.train_file)
            train_dataset = load_dataset(
                "csv", data_files=args.train_file, delimiter="\t"
            )["train"]
        elif args.train_file.endswith(".csv"):
            train_dataset = load_dataset("csv", data_files=args.train_file)["train"]
        else:
            raise ValueError("Only JSON, JSONL, CSV, and TSV files are supported")

    incontext_dataset = None
    if args.incontext_dataset_file is not None:
        with open(args.incontext_dataset_file, "r") as f:
            incontext_dataset = [json.loads(line) for line in f]
    # Get the templates for demonstration or instruction
    demonstration_template = DEMONSTRATION_TEMPLATE[args.task_name]
    # Get all the samples iteratively
    # Generate prompts for all the samples
    task_instruction = TASK_INSTRUCTIONS[args.task_name] 
    all_prompts = get_generation_prompts(
        args,
        train_dataset,
        task_instruction,
        demonstration_template,
        incontext_dataset,
    )
    output_dir = os.path.join(args.output_dir, cur_run_dir)
    prompt_file_path = os.path.join(output_dir, "prompts.jsonl")
    response_file_path = os.path.join(output_dir, "api_responses.jsonl")
    metadata_file_path = os.path.join(output_dir, "metadata.json")
    if args.delete_old_responses:
        if os.path.exists(response_file_path):
            os.remove(response_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # format the prompts into the OpenAI format as a jsonl file in the data directory
    with open(prompt_file_path, "w") as f, open(metadata_file_path, "w") as metadata_f:
        for prompt in all_prompts:
            if args.model == "gpt-3.5-turbo":
                formatted_request = {
                    "model": args.model,
                    "messages": [
                        {"role": "user", "content": f'{prompt["prompt"]} '},
                    ],
                    "temperature": args.temperature,
                    "max_tokens": 600,
                    "top_p": args.top_p,
                    "frequency_penalty": args.frequency_penalty,
                    "presence_penalty": args.presence_penalty,
                }
            else:
                formatted_request = {
                    "model": args.model,
                    "prompt": prompt["prompt"],
                    "temperature": args.temperature,
                    "max_tokens": 600,
                    "top_p": args.top_p,
                    "frequency_penalty": args.frequency_penalty,
                    "presence_penalty": args.presence_penalty,
                }
            metadata = {
                "id": prompt["id"],
                "sample": prompt["sample"],
                "demos": prompt["demos"],
                "task_instruction": prompt["task_instruction"],
            }
            f.write(json.dumps(formatted_request))
            f.write("\n")
            metadata_f.write(json.dumps(metadata))
            metadata_f.write("\n")
    # Set the request url based o whether we are using a chat-based model
    if args.model == "gpt-3.5-turbo":
        request_url = "https://api.openai.com/v1/chat/completions"
    else:
        request_url = "https://api.openai.com/v1/completions"
    # Make API calls
    asyncio.run(
        process_api_requests_from_file(
            prompt_file_path,
            response_file_path,
            request_url,
            api_key,
            args.max_requests_per_minute,
            args.max_tokens_per_minute,
            "cl100k_base",
            args.max_attempts,
            logging.DEBUG,
            metadata_file_path,
        )
    )
    all_proficiency_metrics = {}
    # process the responses and save them in the data directory
    with open(response_file_path, "r") as f:
        api_responses = [json.loads(line) for line in f]
        generated_samples = []
        for api_response in api_responses:
            metadata = api_response[2]
            api_response = api_response[1]
            if "choices" not in api_response:
                if "error" in api_response:
                    print(api_response["error"])
                    print("error in response, generation not available for this prompt")
                else:
                    print("unknown error in response")
                raise ValueError("error in response")
                # cur_sample["generation"] = ""
                # cur_sample["input_to_model"] = metadata["sample"]
                # cur_sample["id"] = metadata["id"]
                # generated_samples.append(cur_sample)
                # continue
            cur_sample = train_dataset[metadata["id"]]
            if args.model != "gpt-3.5-turbo":
                response_text = api_response["choices"][0]["text"]
            else:
                response_text = api_response["choices"][0]["message"]["content"]
            response_text = response_text.strip()
            cur_sample["generation"] = response_text
            cur_sample["input_to_model"] = metadata["sample"]
            cur_sample["id"] = metadata["id"]
            proficiency_metrics = get_proficiency_score(
                response_text, cur_sample, args.task_name
            )
            for metric_name, metric_value in proficiency_metrics.items():
                cur_sample[f"generation_{metric_name}"] = metric_value
                if metric_name not in all_proficiency_metrics:
                    all_proficiency_metrics[metric_name] = []
                all_proficiency_metrics[metric_name].append(metric_value)
            generated_samples.append(cur_sample)
    # sort the samples by the id
    generated_samples = sorted(generated_samples, key=lambda x: x["id"])
    # write the generations to a jsonl file
    with open(
        os.path.join(
            args.output_dir,
            cur_run_dir,
            f"generations.jsonl",
        ),
        "w",
    ) as f:
        assert len(generated_samples) == len(all_prompts) == len(train_dataset)
        # write the generations by appending to the train dataset file
        for sample in generated_samples:
            f.write(json.dumps(sample))
            f.write("\n")
    # write the proficiency scores to a json file
    with open(
        os.path.join(
            args.output_dir,
            cur_run_dir,
            f"proficiency_scores.json",
        ),
        "w",
    ) as f:
        all_proficiency_metrics = {
            metric_name: np.mean(metric_values)
            for metric_name, metric_values in all_proficiency_metrics.items()
        }
        json.dump(all_proficiency_metrics, f)


if __name__ == "__main__":
    main()
