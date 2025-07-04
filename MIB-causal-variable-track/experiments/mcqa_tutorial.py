# # A Causal Model of Simple Multiple Choice Question Answering
#
#

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "CausalAbstraction"))

import random

import torch
from causal.causal_model import CausalModel
from experiments.residual_stream_experiment import PatchResidualStream
from neural.pipeline import LMPipeline
from tasks.simple_MCQA.simple_MCQA import (
    get_causal_model,
    get_counterfactual_datasets,
    get_token_positions,
)

OBJECT_COLORS = [
    ("banana", "yellow"),
    ("grass", "green"),
    ("strawberry", "red"),
    ("coconut", "brown"),
    ("eggplant", "purple"),
    ("blueberry", "blue"),
    ("carrot", "orange"),
    ("coal", "black"),
    ("snow", "white"),
    ("ivory", "white"),
    ("cauliflower", "white"),
    ("bubblegum", "pink"),
    ("lemon", "yellow"),
    ("lime", "green"),
    ("ruby", "red"),
    ("chocolate", "brown"),
    ("emerald", "green"),
    ("sapphire", "blue"),
    ("pumpkin", "orange"),
]
OBJECTS, COLORS = zip(*OBJECT_COLORS)

NUM_CHOICES = 2
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TEMPLATES = [
    "The <object> is <color>. What color is the <object>?"
    + "".join([f"\n<symbol{str(i)}>. <choice{str(i)}>" for i in range(NUM_CHOICES)])
    + "\nAnswer:"
]

variables = (
    ["template", "object_color", "raw_input"]
    + ["symbol" + str(x) for x in range(NUM_CHOICES)]
    + ["choice" + str(x) for x in range(NUM_CHOICES)]
    + ["answer_position", "answer", "raw_output"]
)

values = {"choice" + str(x): COLORS for x in range(NUM_CHOICES)}
values.update({"symbol" + str(x): ALPHABET for x in range(NUM_CHOICES)})
values.update({"answer_position": range(NUM_CHOICES), "answer": ALPHABET})
values.update({"template": TEMPLATES})
values.update({"object_color": OBJECT_COLORS})
values.update({"raw_input": None, "raw_output": None})

parents = {
    "template": [],
    "object_color": [],
    "raw_input": ["template", "object_color"]
    + ["symbol" + str(x) for x in range(NUM_CHOICES)]
    + ["choice" + str(x) for x in range(NUM_CHOICES)],
    "answer_position": ["object_color"]
    + ["choice" + str(x) for x in range(NUM_CHOICES)],
    "answer": ["answer_position"] + ["symbol" + str(x) for x in range(NUM_CHOICES)],
    "raw_output": ["answer"],
}
parents.update({"choice" + str(x): [] for x in range(NUM_CHOICES)})
parents.update({"symbol" + str(x): [] for x in range(NUM_CHOICES)})


def fill_template(*args):
    template, object_color = args[0], args[1]
    symbols = args[2 : 2 + NUM_CHOICES]
    choices = args[2 + NUM_CHOICES : 2 + 2 * NUM_CHOICES]

    object_name, color = object_color
    filled_template = template.replace("<object>", object_name).replace(
        "<color>", color
    )
    for i, symbol in enumerate(symbols):
        filled_template = filled_template.replace(f"<symbol{i}>", symbol)
    for i, choice in enumerate(choices):
        filled_template = filled_template.replace(f"<choice{i}>", choice)
    return filled_template


def get_answer_position(object_color, *choices):
    for i, choice in enumerate(choices):
        if choice == object_color[1]:
            return i


def get_answer(answer_position, *symbols):
    if answer_position is None:
        return None
    return " " + symbols[answer_position]


mechanisms = {
    "template": lambda: random.choice(TEMPLATES),
    "object_color": lambda: random.choice(OBJECT_COLORS),
    **{f"symbol{i}": lambda: random.choice(ALPHABET) for i in range(NUM_CHOICES)},
    **{f"choice{i}": lambda: random.choice(COLORS) for i in range(NUM_CHOICES)},
    "raw_input": fill_template,
    "answer_position": get_answer_position,
    "answer": get_answer,
    "raw_output": lambda x: x,
}

MCQA_causal_model = CausalModel(
    variables, values, parents, mechanisms, id=f"{NUM_CHOICES}_answer_MCQA"
)


def sample_answerable_question():
    input = MCQA_causal_model.sample_input()
    # sample unique choices and symbols
    choices = random.sample(COLORS, NUM_CHOICES)
    symbols = random.sample(ALPHABET, NUM_CHOICES)
    for idx in range(NUM_CHOICES):
        input["choice" + str(idx)] = choices[idx]
        input["symbol" + str(idx)] = symbols[idx]
    if input["object_color"][1] not in [
        input["choice" + str(x)] for x in range(NUM_CHOICES)
    ]:
        index = random.randint(0, NUM_CHOICES - 1)
        input["choice" + str(index)] = input["object_color"][1]
    return input


example = sample_answerable_question()
full_setting = MCQA_causal_model.run_forward(example)
print("Example input:")
for k, v in example.items():
    print(f"{k}: {v}")
print("\nFull setting:")
for k, v in full_setting.items():
    print(f"{k}: {v}")


# # Constructing Counterfactual Datasets for a Causal Analysis

from causal.causal_model import CounterfactualDataset


def same_symbol_different_position():
    input = sample_answerable_question()
    counterfactual = input.copy()

    pos = MCQA_causal_model.run_forward(input)["answer_position"]
    new_pos = random.choice([i for i in range(NUM_CHOICES) if i != pos])
    counterfactual["choice" + str(pos)] = input["choice" + str(new_pos)]
    counterfactual["choice" + str(new_pos)] = input["choice" + str(pos)]
    counterfactual["symbol" + str(pos)] = input["symbol" + str(new_pos)]
    counterfactual["symbol" + str(new_pos)] = input["symbol" + str(pos)]
    input["raw_input"] = MCQA_causal_model.run_forward(input)["raw_input"]
    counterfactual["raw_input"] = MCQA_causal_model.run_forward(counterfactual)[
        "raw_input"
    ]
    return {"input": input, "counterfactual_inputs": [counterfactual]}


def different_symbol_same_position():
    input = sample_answerable_question()
    counterfactual = input.copy()
    current_symbols = [input["symbol" + str(i)] for i in range(NUM_CHOICES)]
    complement = [x for x in ALPHABET if x not in current_symbols]
    # choose from the complement without replacement
    new_symbols = random.sample(complement, NUM_CHOICES)
    for i in range(NUM_CHOICES):
        counterfactual["symbol" + str(i)] = new_symbols[i]
    input["raw_input"] = MCQA_causal_model.run_forward(input)["raw_input"]
    counterfactual["raw_input"] = MCQA_causal_model.run_forward(counterfactual)[
        "raw_input"
    ]
    return {"input": input, "counterfactual_inputs": [counterfactual]}


def different_symbol_different_position():
    input = sample_answerable_question()
    counterfactual = input.copy()

    # Different position
    pos = MCQA_causal_model.run_forward(input)["answer_position"]
    new_pos = random.choice([i for i in range(NUM_CHOICES) if i != pos])
    counterfactual["choice" + str(pos)] = input["choice" + str(new_pos)]
    counterfactual["choice" + str(new_pos)] = input["choice" + str(pos)]

    # Different symbol
    current_symbols = [input["symbol" + str(i)] for i in range(NUM_CHOICES)]
    complement = [x for x in ALPHABET if x not in current_symbols]
    new_symbols = random.sample(complement, NUM_CHOICES)
    for i in range(NUM_CHOICES):
        counterfactual["symbol" + str(i)] = new_symbols[i]

    input["raw_input"] = MCQA_causal_model.run_forward(input)["raw_input"]
    counterfactual["raw_input"] = MCQA_causal_model.run_forward(counterfactual)[
        "raw_input"
    ]
    return {"input": input, "counterfactual_inputs": [counterfactual]}


def random_counterfactual():
    input, counterfactual = sample_answerable_question(), sample_answerable_question()
    input["raw_input"] = MCQA_causal_model.run_forward(input)["raw_input"]
    counterfactual["raw_input"] = MCQA_causal_model.run_forward(counterfactual)[
        "raw_input"
    ]
    return {"input": input, "counterfactual_inputs": [counterfactual]}


counterfactual_datasets = {
    "same_symbol_different_position": CounterfactualDataset.from_sampler(
        100, same_symbol_different_position
    ),
    "different_symbol_different_position": CounterfactualDataset.from_sampler(
        100, different_symbol_different_position
    ),
    "different_symbol_same_position": CounterfactualDataset.from_sampler(
        100, different_symbol_same_position
    ),
    "random_counterfactual": CounterfactualDataset.from_sampler(
        100, random_counterfactual
    ),
}

for name, dataset in counterfactual_datasets.items():
    example = dataset[0]
    print(f"\nCounterfactual dataset: {name}")
    print("\nExample input:")
    for k, v in example["input"].items():
        print(f"{k}: {v}")
    print("\nCounterfactual example:")
    for k, v in example["counterfactual_inputs"][0].items():
        print(f"{k}: {v}")
    print("--" * 20)

# # Loading in a Language Model

device = "cuda" if torch.cuda.is_available() else "cpu"
# model_name = "EleutherAI/pythia-410m"
# model_name = "google/gemma-2-2b-it"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# model_name = "google/gemma-2-2b-it"
# model_name = "microsoft/Phi-4-mini-instruct"

pipeline = LMPipeline(model_name, max_new_tokens=1, device=device, dtype=torch.float16)
pipeline.tokenizer.padding_side = "left"
print("DEVICE:", pipeline.model.device)

# Get a sample input and check model's prediction
sampled_example = next(iter(counterfactual_datasets.values()))[0]
print("INPUT:", sampled_example["input"])
print(
    "EXPECTED OUTPUT:",
    MCQA_causal_model.run_forward(sampled_example["input"])["raw_output"],
)
print("MODEL PREDICTION:", pipeline.dump(pipeline.generate(sampled_example["input"])))


# # Filtering Examples the Language Model Fails On

# from experiments.filter_experiment import FilterExperiment


# def checker(neural_output, causal_output):
#     return neural_output in causal_output


# # Filter the datasets based on model performance
# print("\nFiltering datasets based on model performance...")
# exp = FilterExperiment(pipeline, MCQA_causal_model, checker)
# filtered_datasets = exp.filter(counterfactual_datasets, verbose=True, batch_size=128)

# # Definition token positions of interest

import re

from neural.LM_units import TokenPosition, get_last_token_index


def get_correct_symbol_index(input, pipeline, causal_model):
    """
    Find the index of the correct answer symbol in the prompt.

    Args:
        input (Dict): The input dictionary to a causal model
        pipeline: The tokenizer pipeline

    Returns:
        list[int]: List containing the index of the correct answer symbol token
    """
    # Run the model to get the answer position
    output = causal_model.run_forward(input)
    pos = output["answer_position"]
    correct_symbol = output[f"symbol{pos}"]
    prompt = input["raw_input"]

    # Find all single uppercase letters in the prompt
    matches = list(re.finditer(r"\b[A-Z]\b", prompt))

    # Find the match corresponding to our correct symbol
    symbol_match = None
    for match in matches:
        if prompt[match.start() : match.end()] == correct_symbol:
            symbol_match = match
            break

    if not symbol_match:
        raise ValueError(
            f"Could not find correct symbol {correct_symbol} in prompt: {prompt}"
        )

    # Get the substring up to the symbol match end
    substring = prompt[: symbol_match.end()]
    tokenized_substring = list(pipeline.load(substring)["input_ids"][0])

    # The symbol token will be at the end of the substring
    return [len(tokenized_substring) - 1]


# Create TokenPosition object
token_positions = [
    TokenPosition(
        lambda x: get_correct_symbol_index(x, pipeline, MCQA_causal_model),
        pipeline,
        id="correct_symbol",
    ),
    TokenPosition(
        lambda x: [get_correct_symbol_index(x, pipeline, MCQA_causal_model)[0] + 1],
        pipeline,
        id="correct_symbol_period",
    ),
    TokenPosition(
        lambda x: get_last_token_index(x, pipeline), pipeline, id="last_token"
    ),
]

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
# model_name = "EleutherAI/pythia-410m"
pipeline = LMPipeline(model_name, max_new_tokens=1, device=device, dtype=torch.float16)
pipeline.tokenizer.padding_side = "left"

# Data and model
counterfactual_datasets = get_counterfactual_datasets(
    hf=True, size=None, load_private_data=False
)
causal_model = get_causal_model()
token_positions = get_token_positions(pipeline, causal_model)


# Checker function
def checker(neural_output, causal_output):
    return neural_output in causal_output


# DAS config
config = {
    "batch_size": 16,
    "evaluation_batch_size": 128,
    "training_epoch": 8,
    "n_features": 16,
    "temperature_schedule": (0.3, 0.3),
    "regularization_coefficient": 0.1,
    "intervenable_model_kwargs": {
        "start_temperature": 0.3,
        "learnable_temperature": False,
        "straight_through": False,
        "inference_binarization": True,
        "eps": 1e-6,
        "threshold": 0.5,
        "stochastic": True,
    },
}

# PatchResidualStream experiment
experiment = PatchResidualStream(
    pipeline=pipeline,
    causal_model=causal_model,
    layers=list(range(pipeline.get_num_layers())),
    # layers=[1],
    token_positions=token_positions,
    checker=checker,
    config=config,
)

experiment.train_interventions(
    counterfactual_datasets, ["answer_pointer"], method="DBM", verbose=True
)

raw_results = experiment.perform_interventions(
    counterfactual_datasets,
    verbose=True,
    target_variables_list=[["answer_pointer"]],
    save_dir="logs/mcqa_tutorial_dbm",
)

print("\nHeatmaps for 'answer_position' variable:")
experiment.plot_heatmaps(
    raw_results, ["answer_pointer"], save_path="logs/mcqa_tutorial_dbm"
)
