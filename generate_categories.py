import re

# Insert code for taking arguments from command line
import argparse
from datasets import load_dataset
import openai
import os
import json
import asyncio
from api_request_parallel_processor import process_api_requests_from_file
from utils.misc_utils import authenticate, get_prompt
from utils.args import add_args
import random
import inflect
from tqdm import tqdm

inflect_engine = inflect.engine()

# Import global variables
from utils.templates import (
    DEMONSTRATION_TEMPLATE,
    TASK_INSTRUCTIONS,
    CATEGORY_GENERATION_PROMPTS,
)

def generate_categories(
    args, train_dataset, demonstration_template, task_instruction, category
):
    category_prompts = CATEGORY_GENERATION_PROMPTS[args.task_name]
    # loop through in-context samples, along with the task instruction

    prompt_file_path = os.path.join(args.output_dir, f"prompts_{category}.jsonl")
    response_file_path = os.path.join(
        args.output_dir, f"api_responses_{category}.jsonl"
    )
    parsed_response_file_path = os.path.join(
        args.output_dir, f"api_generated_dataset_{category}.jsonl"
    )
    metadata_file_path = os.path.join(args.output_dir, f"api_metadata_{category}.jsonl")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.delete_old_responses:
        if os.path.exists(response_file_path):
            os.remove(response_file_path)

    prompts = []
    for _ in range(len(train_dataset) // args.few_shot_k):
        if args.incontext_sampling_strategy == "random":
            incontext_train = train_dataset.select(
                random.sample(range(len(train_dataset)), args.few_shot_k)
            )
        elif args.incontext_sampling_strategy == "fixed":
            if len(train_dataset) > args.few_shot_k:
                print(
                    "Warning: the number of samples in the train dataset is greater than k. Truncating up to k samples"
                )
                incontext_train = train_dataset[: args.few_shot_k]
            else:
                print(
                    "Warning: the number of samples in the train dataset is less than k."
                )
                incontext_train = train_dataset[:]
        else:
            raise ValueError(
                "incontext_sampling_strategy must be either 'random' or 'fixed'"
            )
        _, collated_demos = get_prompt(
            args, incontext_train, "", demonstration_template
        )
        cur_prompt = (
            collated_demos
            + f"*Task Instruction*:{task_instruction}"
            + category_prompts[category]
        )
        prompts.append(
            {
                "prompt": cur_prompt,
                "type": category,
                "task_instruction": task_instruction,
                "collated_demos": collated_demos,
            }
        )
    # use the collated demos to find skills
    # format the prompts into the OpenAI format as a jsonl file in the data directory
    with open(prompt_file_path, "w") as f, open(metadata_file_path, "w") as metadata_f:
        for prompt in prompts:
            if args.model == "gpt-3.5-turbo":
                formatted_request = {
                    "model": "gpt-3.5-turbo-16k",
                    "messages": [
                        {"role": "user", "content": prompt["collated_demos"]},
                        {
                            "role": "user",
                            "content": f"Understand and note these {args.few_shot_k} examples. Also understand the original task for these examples. Please note that the examples are not exhaustive.",
                        },
                        {
                            "role": "user",
                            "content": f"Good Job, The original task instruction is: {prompt['task_instruction']}. {category_prompts[category]}",
                        },
                    ],
                    "temperature": args.temperature,
                    "max_tokens": 1700,
                    "top_p": args.top_p,
                    "frequency_penalty": args.frequency_penalty,
                    "presence_penalty": args.presence_penalty,
                }
            metadata = {
                "type": prompt["type"],
                "task_instruction": prompt["task_instruction"],
            }
            f.write(json.dumps(formatted_request))
            f.write("\n")
            metadata_f.write(json.dumps(metadata))
            metadata_f.write("\n")

    # Set the request url based on whether we are using a chat-based model
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
            args.api_key,
            args.max_requests_per_minute,
            args.max_tokens_per_minute,
            "cl100k_base",
            args.max_attempts,
            args.logging_level,
            metadata_file_path,
        )
    )
    # process the responses and save them in the data directory
    regex_template = r"({}:|{}:|\d+.)(?P<category>.+)".format(
        category, category.capitalize()
    )
    with open(response_file_path, "r") as f:
        api_responses = [json.loads(line) for line in f]
        generated_samples = []
        for api_response in api_responses:
            metadata = api_response[2]
            api_response = api_response[1]
            if "error" in api_response:
                print(api_response["error"])
                continue
            # parse the response with regex filtering
            if "choices" not in api_response or len(api_response["choices"]) == 0:
                continue
            if args.model == "gpt-3.5-turbo":
                response_text = api_response["choices"][0]["message"]["content"]
            else:
                response_text = api_response["choices"][0]["text"]
            response_examples = response_text.split("\n")
            response_examples = list(filter(lambda x: x != "", response_examples))
            # Keep track of the number of examples that failed to parse
            failed_to_match = 0
            total_examples = 0
            # parse regex with this template
            for response_example in response_examples:
                total_examples += 1
                # Split the response into individual keys
                match = re.match(regex_template, response_example)
                # Extract the parsed values from the match object
                if (
                    match
                    and "category" in match.groupdict()
                    # and "explanation" in match.groupdict()
                ):
                    generated_category = match.group("category")
                    # generated_explanation = match.group("explanation")
                    output_example = {
                        category: generated_category,
                        # "explanation": generated_explanation,
                        **metadata,
                    }
                    generated_samples.append(output_example)
                else:
                    print("Failed to match example: {}".format(response_example))
                    failed_to_match += 1
            # Print the number of examples that failed to parse
            print(
                "Failed to match {} out of {} examples ({} percent)".format(
                    failed_to_match / 2,
                    total_examples,
                    (failed_to_match / 2) / total_examples * 100,
                )
            )
    # call LLM again to clean up the generated samples
    NUM_PRUNE_CATEGORIES = args.num_categories
    pruned_generated_samples = generated_samples
    num_categories_in_chunk = args.num_categories_pruning_factor * args.num_categories
    while len(pruned_generated_samples) > num_categories_in_chunk:
        cur_pruned_categories = []
        generated_samples_chunks = [
            pruned_generated_samples[i : i + num_categories_in_chunk]
            for i in range(0, len(pruned_generated_samples), num_categories_in_chunk)
        ]
        for generated_samples_chunk in tqdm(generated_samples_chunks):
            if len(generated_samples_chunk) < NUM_PRUNE_CATEGORIES:
                cur_pruned_categories.extend(generated_samples_chunk)
                continue
            all_categories = (
                f"{category.capitalize()}: "
                + f"\n{category.capitalize()}: ".join(
                    [sample[category] for sample in generated_samples_chunk]
                )
            )
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",
                    messages=[
                        {"role": "user", "content": all_categories},
                        {
                            "role": "user",
                            "content": f"These are the {inflect_engine.plural_noun(category)} for the following task instruction: {task_instruction}. Understand the {inflect_engine.plural_noun(category)} and the task.",
                        },
                        {
                            "role": "user",
                            "content": f"Good Job, now generate a concise list of UPTO {NUM_PRUNE_CATEGORIES} {inflect_engine.plural_noun(category)} from the given {inflect_engine.plural_noun(category)}, which are critically relevant. Structure the format as <{category.capitalize()}>:<one-line explanation of {category}>. Generate a numbered list.",
                        },
                    ],
                    temperature=args.temperature,
                    max_tokens=1700,
                    top_p=args.top_p,
                    frequency_penalty=args.frequency_penalty,
                    presence_penalty=args.presence_penalty,
                )
            except Exception as e:
                print(e)
                continue
            if (
                "error" in response
                or "choices" not in response
                or len(response["choices"]) == 0
            ):
                continue
            response_text = response["choices"][0]["message"]["content"]
            response_examples = response_text.split("\n")
            response_examples = list(filter(lambda x: x != "", response_examples))
            # Keep track of the number of examples that failed to parse
            failed_to_match = 0
            total_examples = 0
            # parse regex with this template
            for response_example in response_examples:
                total_examples += 1
                # Split the response into individual keys
                match = re.match(regex_template, response_example)
                # Extract the parsed values from the match object
                if match and "category" in match.groupdict():
                    generated_category = match.group("category")
                    output_example = {
                        category: generated_category,
                        **metadata,
                    }
                    cur_pruned_categories.append(output_example)
                else:
                    print("Failed to match example: {}".format(response_example))
                    failed_to_match += 1
            # Print the number of examples that failed to parse
            print(
                "Failed to match {} out of {} examples ({} percent)".format(
                    failed_to_match / 2,
                    total_examples,
                    (failed_to_match / 2) / total_examples * 100,
                )
            )
        pruned_generated_samples = cur_pruned_categories

    # prune the final list of categories with another call ot the LLM
    cur_pruned_categories = []
    all_categories = f"{category.capitalize()}: " + f"\n{category.capitalize()}: ".join(
        [sample[category] for sample in pruned_generated_samples]
    )
    try:
        response = openai.ChatCompletion.create(
            model=args.model,
            messages=[
                {"role": "user", "content": all_categories},
                {
                    "role": "user",
                    "content": f"These are the {inflect_engine.plural_noun(category)} for the following task instruction: {task_instruction}. Understand the {inflect_engine.plural_noun(category)} and the task.",
                },
                {
                    "role": "user",
                    "content": f"Good Job, now generate a concise list of UPTO {args.num_categories} {inflect_engine.plural_noun(category)} from the given {inflect_engine.plural_noun(category)}, which are critically relevant. Structure the format as <{category.capitalize()}>:<one-line explanation of {category}>. Generate a numbered list. [IMPORTANT] Please avoid repetition and duplicatiom of {inflect_engine.plural_noun(category)} and generate distinct {inflect_engine.plural_noun(category)}.",
                },
            ],
            temperature=args.temperature,
            max_tokens=1700,
            top_p=args.top_p,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
        )
    except:
        print(
            "exception encountered while creating pruned set of categories, skipping this iteration"
        )
        return
    if (
        "error" in response
        or "choices" not in response
        or len(response["choices"]) == 0
    ):
        return
    response_text = response["choices"][0]["message"]["content"]
    response_examples = response_text.split("\n")
    response_examples = list(filter(lambda x: x != "", response_examples))
    # Keep track of the number of examples that failed to parse
    failed_to_match = 0
    total_examples = 0
    # parse regex with this template
    for response_example in response_examples:
        total_examples += 1
        # Split the response into individual keys
        match = re.match(regex_template, response_example)
        # Extract the parsed values from the match object
        if match and "category" in match.groupdict():
            generated_category = match.group("category")
            output_example = {
                category: generated_category,
                **metadata,
            }
            cur_pruned_categories.append(output_example)
        else:
            print("Failed to match example: {}".format(response_example))
            failed_to_match += 1
    pruned_generated_samples = cur_pruned_categories
    # Print the number of examples that failed to parse
    print(
        "Failed to match {} out of {} examples ({} percent)".format(
            failed_to_match / 2,
            total_examples,
            (failed_to_match / 2) / total_examples * 100,
        )
    )
    with open(parsed_response_file_path, "w") as f:
        for sample in pruned_generated_samples:
            f.write(json.dumps(sample))
            f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        type=str,
        default="subtask",
        choices=["domain", "subtask"],
        help="The type of category to generate",
    )
    parser.add_argument(
        "--num_categories",
        type=int,
        default=15,
        help="The number of categories to generate",
    )

    parser.add_argument(
        "--num_categories_pruning_factor",
        type=int,
        default=4,
        help="How aggressively to shrink the large list of generated categories into num_categories number of categories",
    )
    parser = add_args(parser)
    args = parser.parse_args()
    api_key = authenticate(args)
    args.api_key = api_key
    demonstration_template = DEMONSTRATION_TEMPLATE[args.task_name]
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

    # Check the type of entity
    task_instruction = TASK_INSTRUCTIONS[args.task_name]
    generate_categories(
        args,
        train_dataset,
        demonstration_template,
        task_instruction,
        args.category,
    )


if __name__ == "__main__":
    main()
