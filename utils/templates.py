PROFICIENCY_METRICS = {
    "mbpp": ["pass@1"],
    "knkarthick_dialogsum": ["rouge1", "rouge2", "rougeL"],
    "mmlu_biology": ["accuracy"],
    "medmcqa": ["accuracy"],
}
TASK_INSTRUCTIONS = {
    "mbpp": "Given the natural language description of a python function, and the associated test cases, generate the python function.",
    "knkarthick_dialogsum": "Given the following daily conversations, briefly summarize the conversaton.",
    "mmlu_biology": "The following are multiple choice questions about clinical biology. Each question has 4 answer choices. Select the right answer.",
    "medmcqa": "The following are multiple choice questions across a wide range of medical subjects and topics. Each question has 4 answer choices. Select the right answer.",
}
DEMONSTRATION_TEMPLATE = {
    "mbpp": "Function description: {prompt} Test Cases: {test_list} Code: {code}",
    "knkarthick_dialogsum": "Dialogue: {dialogue} Summary: {summary}",
    "mmlu_biology": "Question: {question} A: {A} B:{B} C:{C} D:{D} Answer: {answer}",
    "medmcqa": "Question: {question} A: {opa} B:{opb} C:{opc} D:{opd} Answer: {answer}",
}
LABEL_KEY = {
    "mbpp": "code",
    "knkarthick_dialogsum": "summary",
    "mmlu_biology": "answer",
    "medmcqa": "answer",
}


CLASSIFICATION_TASKS = ["mmlu_biology", "medmcqa"]


SCORING_PROMPTS = {}
SCORING_PROMPTS["mbpp"] = {
    "domain": "Given the input and output from a language model, Rate to what degree the input and output belong to each of the following domains. Rate on a scale of 1-5, with 5 being compeletely belongs and 1 being not belonging at all. \n. [Important] For each domain, format the output as, [Domain 1: <domain>, Score: <score>, Evidence: <Evidence for score>] 'n' [Domain 2: <domain>, Score: <score>, Evidence: <Evidence for score>] 'n' [Domain N: <domain>, Score: <score>, Evidence: <Evidence for score>]. [Important] Make sure to include concrete evidence based on the input and the output to JUSTIFY the score. Remember you are an ACCURATE, FAITHFUL, CRITICAL and FAIR judge \n'.'",
    "subtask": "Given the input and output from a language model, Rate to what degree each of the following subtasks are used to generate the output from the input. Rate on a scale of 1-5, with 5 being very used and 1 being not used at all. \n. [Important] For each subtask, format the output as [Subtask 1: <subtask>, Score: <score>, Evidence: <Evidence for score>] 'n' [Subtask 2: <subtask>, Score: <score>, Evidence: <Evidence for score>] 'n' [Subtask N: <subtask>,Score: <score>; Evidence: <Evidence for score>]. [IMPORTANT] Do NOT add '\n' between subtask, score and explanation. [Important] Make sure to include concrete evidence based on the input and the output to JUSTIFY the score. Remember you are an ACCURATE, FAITHFUL, CRITICAL and FAIR judge \n'.'",
}
SCORING_PROMPTS["knkarthick_dialogsum"] = {
    "domain": "Given the input and output from a language model, Rate to what degree the input and output belong to each of the following domains. Rate on a scale of 1-5, with 5 being compeletely belongs and 1 being not belonging at all. \n. [Important] For each domain, format the output as, [Domain 1: <domain>, Score: <score>, Evidence: <Evidence for score>] 'n' [Domain 2: <domain>, Score: <score>, Evidence: <Evidence for score>] 'n' [Domain N: <domain>, Score: <score>, Evidence: <Evidence for score>]. [Important] Make sure to include concrete evidence based on the input and the output to JUSTIFY the score. Remember you are an ACCURATE, FAITHFUL, CRITICAL and FAIR judge \n'.'",
    "subtask": "Given the input and output from a language model, Rate to what degree each of the following subtasks are used to generate the output from the input. Rate on a scale of 1-5, with 5 being very used and 1 being not used at all. \n. [Important] For each subtask, format the output as [Subtask 1: <subtask>, Score: <score>, Evidence: <Evidence for score>] 'n' [Subtask 2: <subtask>, Score: <score>, Evidence: <Evidence for score>] 'n' [Subtask N: <subtask>,Score: <score>; Evidence: <Evidence for score>]. [IMPORTANT] Do NOT add '\n' between subtask, score and explanation. [Important] Make sure to include concrete evidence based on the input and the output to JUSTIFY the score. Remember you are an ACCURATE, FAITHFUL, CRITICAL and FAIR judge \n'.'",
}
SCORING_PROMPTS["mmlu_biology"] = {
    "domain": "Given the input to a language model, Rate to what degree the input belong to each of the following domains. Rate on a scale of 1-5, with 5 being compeletely belongs and 1 being not belonging at all. \n. [Important] For each domain, format the output as, [Domain 1: <domain>, Score: <score>, Evidence: <Evidence for score>] 'n' [Domain 2: <domain>, Score: <score>, Evidence: <Evidence for score>] 'n' [Domain N: <domain>, Score: <score>, Evidence: <Evidence for score>]. [Important] Make sure to include concrete evidence based on the input to JUSTIFY the score. Remember you are an ACCURATE, FAITHFUL, CRITICAL and FAIR judge \n'.'",
    "subtask": "Given the input to a language model, Rate to what degree each of the following subtasks are needed to successfully understand and complete the task. Rate on a scale of 1-5, with 5 being very used and 1 being not used at all. \n. [Important] For each subtask, format the output as [Subtask 1: <subtask>, Score: <score>, Evidence: <Evidence for score>] 'n' [Subtask 2: <subtask>, Score: <score>, Evidence: <Evidence for score>] 'n' [Subtask N: <subtask>,Score: <score>; Evidence: <Evidence for score>]. [IMPORTANT] Do NOT add '\n' between subtask, score and explanation. [Important] Make sure to include concrete evidence based on the input to JUSTIFY the score. Remember you are an ACCURATE, FAITHFUL, CRITICAL and FAIR judge \n'.'",
}
SCORING_PROMPTS["medmcqa"] = {
    "domain": "Given the input to a language model, Rate to what degree the input belong to each of the following domains. Rate on a scale of 1-5, with 5 being compeletely belongs and 1 being not belonging at all. \n. [Important] For each domain, format the output as, [Domain 1: <domain>, Score: <score>, Evidence: <Evidence for score>] 'n' [Domain 2: <domain>, Score: <score>, Evidence: <Evidence for score>] 'n' [Domain N: <domain>, Score: <score>, Evidence: <Evidence for score>]. [Important] Make sure to include concrete evidence based on the input to JUSTIFY the score. Remember you are an ACCURATE, FAITHFUL, CRITICAL and FAIR judge \n'.'",
    "subtask": "Given the input to a language model, Rate to what degree each of the following subtasks are needed to successfully understand and complete the task. Rate on a scale of 1-5, with 5 being very used and 1 being not used at all. \n. [Important] For each subtask, format the output as [Subtask 1: <subtask>, Score: <score>, Evidence: <Evidence for score>] 'n' [Subtask 2: <subtask>, Score: <score>, Evidence: <Evidence for score>] 'n' [Subtask N: <subtask>,Score: <score>; Evidence: <Evidence for score>]. [IMPORTANT] Do NOT add '\n' between subtask, score and explanation. [Important] Make sure to include concrete evidence based on the input to JUSTIFY the score. Remember you are an ACCURATE, FAITHFUL, CRITICAL and FAIR judge \n'.'",
}

SYSTEM_PROMPTS = {}
SYSTEM_PROMPTS[
    "mbpp"
] = f"You are an ACCURATE, FAITHFUL, CRITICAL and FAIR judge for rating the output from a language model and provide EVIDENCE based on the input and the output. \n"
SYSTEM_PROMPTS[
    "knkarthick_dialogsum"
] = f"You are an ACCURATE, FAITHFUL, CRITICAL and FAIR judge for rating the output from a language model and provide EVIDENCE based on the input and the output. \n"
SYSTEM_PROMPTS[
    "mmlu_biology"
] = f"You are an ACCURATE, FAITHFUL, CRITICAL and FAIR judge. Provide EVIDENCE for your choices. \n"
SYSTEM_PROMPTS[
    "medmcqa"
] = f"You are an ACCURATE, FAITHFUL, CRITICAL and FAIR judge. Provide EVIDENCE for your choices. \n"


CATEGORY_GENERATION_PROMPTS = {
    "mbpp": {
        "domain": "Given the following examples, What are relevant domains for the following programs? Focus on the example programs BUT be general. Structure the response as a numbered list.",
        "subtask": "Given the example programs, What are specific ATOMIC sub-tasks a machine learning model need to be competent at for the underlying task? Focus on the example programs BUT be general. [IMPORTANT] Do NOT list the overall task as a subtask and be GENERAL. Structure the response as: Subtask:. Generate a numbered list.",
    },
    "knkarthick_dialogsum": {
        "domain": "Given the following conversations, What are relevant domains for the data? Focus on the example data BUT be general. Structure the response as a numbered list.",
        "subtask": "Given the example conversations, What are specific sub-tasks a machine learning model need to be competent at for the underlying task? Focus on the example data BUT be general. [IMPORTANT] Do NOT list the overall task as a subtask and be GENERAL. Structure the response as: Subtask:. Generate a numbered list.",
    },
    "mmlu_biology": {
        "domain": "Given the following examples, What are relevant domains for the data? Focus on the example data BUT be general. Structure the response as a numbered list.",
        "subtask": "Given the example questions and answers on clinical biology, What are sub-tasks a machine learning model need to be competent at to be a good medical assistant. Focus on the example data BUT be please be general. For instance. [IMPORTANT] Do NOT list the overall task as a subtask and be GENERAL while being GROUNDED in the example data. Structure the response as: Subtask: <subtask>. Generate a numbered list.",
    },
    "medmcqa": {
        "domain": "Given the following examples, What are relevant domains for the data? Focus on the example data BUT be general. Structure the response as a numbered list.",
        "subtask": "Given the example questions and answers on clinical biology, What are sub-tasks a machine learning model need to be competent at to be a good medical assistant. Focus on the example data BUT be please be general. For instance. [IMPORTANT] Do NOT list the overall task as a subtask and be GENERAL while being GROUNDED in the example data. Structure the response as: Subtask: <subtask>. Generate a numbered list.",
    },
}
