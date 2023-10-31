import logging


def add_args(parser):
    # Dataset Arguments
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Eval file for generations ",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/",
        help="The directory with the input files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/",
        help="The directory to store the generated dataset.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=".env",
        help="OpenAI API key",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="Which OpenAI model to use.",
    )
    parser.add_argument(
        "--few_shot",
        action="store_true",
        help="Run the few-shot baseline instead of the zero-shot baseline.",
    )
    parser.add_argument(
        "--few_shot_k",
        type=int,
        default=5,
        help="The number of examples to use in the prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Temperature in the API call. Higher temperature means more randomness.",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=int,
        default=0.02,
        help="frequency_penalty in the API call.",
    )
    parser.add_argument(
        "--presence_penalty",
        type=int,
        default=0,
        help="presence penalty in the API call.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="top_p in nucleus sampling",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--task_name", type=str, required=True, help="Task name")
    parser.add_argument(
        "--max_num_samples",
        type=int,
        default=100,
        help="maximum number of samples from the train dataset to use for in-context examples",
    )
    parser.add_argument(
        "--incontext_sampling_strategy",
        type=str,
        default="random",
        choices=["random", "fixed", "equal"],
        help="strategy to sample in-context examples from the train dataset. Equal samples equal number of instances from each class. Fixed samples the same number of instances from each class. Random samples randomly from the train dataset.",
    )

    parser.add_argument("--max_requests_per_minute", type=int, default=20)
    parser.add_argument("--max_tokens_per_minute", type=int, default=50_000)
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--logging_level", default=logging.INFO)
    parser.add_argument(
        "--delete_old_responses",
        action="store_true",
        help="Delete old responses.",
    )
    return parser
