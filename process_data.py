import openai
import os
import argparse
from datasets import load_dataset
import logging
import json
from utils.args import add_args
from utils.misc_utils import authenticate, seed_function


def main():
    parser = argparse.ArgumentParser()
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

    else:
        # Downloading and loading a dataset from the hub.
        if args.task_name == "mbpp":
            raw_datasets = load_dataset(args.task_name, "sanitized")
        elif args.task_name == "mmlu_biology":
            raw_datasets_all = load_dataset("cais/mmlu", "clinical_knowledge")
            raw_datasets = {}
            # process the dataset to get the format we want
            for split in raw_datasets_all.keys():
                answer_choices = ["A", "B", "C", "D"]
                raw_datasets_all[split] = raw_datasets_all[split].rename_column(
                    "answer", "answer_index"
                )
                if split == "auxiliary_train":
                    raw_datasets["train"] = raw_datasets_all[split].map(
                        lambda x: {
                            "question": x["question"],
                            "A": x["choices"][0],
                            "B": x["choices"][1],
                            "C": x["choices"][2],
                            "D": x["choices"][3],
                            "answer": answer_choices[x["answer_index"]],
                        }
                    )
                else:
                    raw_datasets[split] = raw_datasets_all[split].map(
                        lambda x: {
                            "question": x["question"],
                            "A": x["choices"][0],
                            "B": x["choices"][1],
                            "C": x["choices"][2],
                            "D": x["choices"][3],
                            "answer": answer_choices[x["answer_index"]],
                        }
                    )
        elif args.task_name == "medmcqa":
            raw_datasets = load_dataset("medmcqa")
            for split in raw_datasets.keys():
                answer_choices = ["A", "B", "C", "D"]
                raw_datasets[split] = raw_datasets[split].map(
                    lambda x: {
                        "question": x["question"],
                        "answer": answer_choices[x["cop"]],
                    }
                )
        else:
            raw_datasets = load_dataset(args.task_name)
        train_dataset = raw_datasets["train"]
        validation_dataset = raw_datasets["validation"]
        if "test" in raw_datasets:
            test_dataset = raw_datasets["test"]
        train_dataset = train_dataset.shuffle(seed=args.seed)
        # Shuffle and preprocess the validation dataset
        validation_dataset = validation_dataset.shuffle(seed=args.seed)
        if "test" in raw_datasets:
            test_dataset = test_dataset.shuffle(seed=args.seed)
        # Save the original dataset
        train_dataset.to_json(
            os.path.join(
                args.output_dir,
                str(args.task_name).replace("/", "_"),
                f"original_{str(args.task_name).replace('/', '_')}.json",
            )
        )
        validation_dataset.to_json(
            os.path.join(
                args.output_dir,
                str(args.task_name).replace("/", "_"),
                f"original_{str(args.task_name).replace('/', '_')}_validation.json",
            )
        )
        if "test" in raw_datasets:
            test_dataset.to_json(
                os.path.join(
                    args.output_dir,
                    str(args.task_name).replace("/", "_"),
                    f"original_{str(args.task_name).replace('/', '_')}_test.json",
                )
            )
    if len(train_dataset) > args.max_num_samples:
        train_dataset = train_dataset.select(range(args.max_num_samples))
    train_dataset.to_json(
        os.path.join(
            args.output_dir,
            str(args.task_name).replace("/", "_"),
            f"original_{str(args.task_name).replace('/', '_')}_truncated_{args.max_num_samples}.json",
        )
    )


if __name__ == "__main__":
    main()
