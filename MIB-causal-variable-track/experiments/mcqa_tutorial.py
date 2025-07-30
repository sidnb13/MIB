# # A Causal Model of Simple Multiple Choice Question Answering
#
#

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "CausalAbstraction"))

import random

import hydra
import torch
from causal.causal_model import CausalModel
from experiments.residual_stream_experiment import PatchResidualStream
from neural.pipeline import LMPipeline
from omegaconf import DictConfig
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

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def create_causal_model(num_choices: int):
    """Create the MCQA causal model with configurable number of choices."""
    templates = [
        "The <object> is <color>. What color is the <object>?"
        + "".join([f"\n<symbol{str(i)}>. <choice{str(i)}>" for i in range(num_choices)])
        + "\nAnswer:"
    ]

    variables = (
        ["template", "object_color", "raw_input"]
        + ["symbol" + str(x) for x in range(num_choices)]
        + ["choice" + str(x) for x in range(num_choices)]
        + ["answer_position", "answer", "raw_output"]
    )

    values = {"choice" + str(x): COLORS for x in range(num_choices)}
    values.update({"symbol" + str(x): ALPHABET for x in range(num_choices)})
    values.update({"answer_position": range(num_choices), "answer": ALPHABET})
    values.update({"template": templates})
    values.update({"object_color": OBJECT_COLORS})
    values.update({"raw_input": None, "raw_output": None})

    parents = {
        "template": [],
        "object_color": [],
        "raw_input": ["template", "object_color"]
        + ["symbol" + str(x) for x in range(num_choices)]
        + ["choice" + str(x) for x in range(num_choices)],
        "answer_position": ["object_color"]
        + ["choice" + str(x) for x in range(num_choices)],
        "answer": ["answer_position"] + ["symbol" + str(x) for x in range(num_choices)],
        "raw_output": ["answer"],
    }
    parents.update({"choice" + str(x): [] for x in range(num_choices)})
    parents.update({"symbol" + str(x): [] for x in range(num_choices)})

    def fill_template(*args):
        template, object_color = args[0], args[1]
        symbols = args[2 : 2 + num_choices]
        choices = args[2 + num_choices : 2 + 2 * num_choices]

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
        "template": lambda: random.choice(templates),
        "object_color": lambda: random.choice(OBJECT_COLORS),
        **{f"symbol{i}": lambda: random.choice(ALPHABET) for i in range(num_choices)},
        **{f"choice{i}": lambda: random.choice(COLORS) for i in range(num_choices)},
        "raw_input": fill_template,
        "answer_position": get_answer_position,
        "answer": get_answer,
        "raw_output": lambda x: x,
    }

    return CausalModel(
        variables, values, parents, mechanisms, id=f"{num_choices}_answer_MCQA"
    )


def sample_answerable_question(causal_model, num_choices: int):
    """Sample an answerable question from the causal model."""
    input = causal_model.sample_input()
    # sample unique choices and symbols
    choices = random.sample(COLORS, num_choices)
    symbols = random.sample(ALPHABET, num_choices)
    for idx in range(num_choices):
        input["choice" + str(idx)] = choices[idx]
        input["symbol" + str(idx)] = symbols[idx]
    if input["object_color"][1] not in [
        input["choice" + str(x)] for x in range(num_choices)
    ]:
        index = random.randint(0, num_choices - 1)
        input["choice" + str(index)] = input["object_color"][1]
    return input


# # Constructing Counterfactual Datasets for a Causal Analysis

from causal.causal_model import CounterfactualDataset


def same_symbol_different_position(causal_model, num_choices: int):
    input = sample_answerable_question(causal_model, num_choices)
    counterfactual = input.copy()

    pos = causal_model.run_forward(input)["answer_position"]
    new_pos = random.choice([i for i in range(num_choices) if i != pos])
    counterfactual["choice" + str(pos)] = input["choice" + str(new_pos)]
    counterfactual["choice" + str(new_pos)] = input["choice" + str(pos)]
    counterfactual["symbol" + str(pos)] = input["symbol" + str(new_pos)]
    counterfactual["symbol" + str(new_pos)] = input["symbol" + str(pos)]
    input["raw_input"] = causal_model.run_forward(input)["raw_input"]
    counterfactual["raw_input"] = causal_model.run_forward(counterfactual)["raw_input"]
    return {"input": input, "counterfactual_inputs": [counterfactual]}


def different_symbol_same_position(causal_model, num_choices: int):
    input = sample_answerable_question(causal_model, num_choices)
    counterfactual = input.copy()
    current_symbols = [input["symbol" + str(i)] for i in range(num_choices)]
    complement = [x for x in ALPHABET if x not in current_symbols]
    # choose from the complement without replacement
    new_symbols = random.sample(complement, num_choices)
    for i in range(num_choices):
        counterfactual["symbol" + str(i)] = new_symbols[i]
    input["raw_input"] = causal_model.run_forward(input)["raw_input"]
    counterfactual["raw_input"] = causal_model.run_forward(counterfactual)["raw_input"]
    return {"input": input, "counterfactual_inputs": [counterfactual]}


def different_symbol_different_position(causal_model, num_choices: int):
    input = sample_answerable_question(causal_model, num_choices)
    counterfactual = input.copy()

    # Different position
    pos = causal_model.run_forward(input)["answer_position"]
    new_pos = random.choice([i for i in range(num_choices) if i != pos])
    counterfactual["choice" + str(pos)] = input["choice" + str(new_pos)]
    counterfactual["choice" + str(new_pos)] = input["choice" + str(pos)]

    # Different symbol
    current_symbols = [input["symbol" + str(i)] for i in range(num_choices)]
    complement = [x for x in ALPHABET if x not in current_symbols]
    new_symbols = random.sample(complement, num_choices)
    for i in range(num_choices):
        counterfactual["symbol" + str(i)] = new_symbols[i]

    input["raw_input"] = causal_model.run_forward(input)["raw_input"]
    counterfactual["raw_input"] = causal_model.run_forward(counterfactual)["raw_input"]
    return {"input": input, "counterfactual_inputs": [counterfactual]}


def random_counterfactual(causal_model, num_choices: int):
    input, counterfactual = (
        sample_answerable_question(causal_model, num_choices),
        sample_answerable_question(causal_model, num_choices),
    )
    input["raw_input"] = causal_model.run_forward(input)["raw_input"]
    counterfactual["raw_input"] = causal_model.run_forward(counterfactual)["raw_input"]
    return {"input": input, "counterfactual_inputs": [counterfactual]}


def create_counterfactual_datasets(causal_model, num_choices: int, dataset_size: int):
    """Create counterfactual datasets for the experiment."""

    def same_symbol_different_position_factory():
        return same_symbol_different_position(causal_model, num_choices)

    def different_symbol_different_position_factory():
        return different_symbol_different_position(causal_model, num_choices)

    def different_symbol_same_position_factory():
        return different_symbol_same_position(causal_model, num_choices)

    def random_counterfactual_factory():
        return random_counterfactual(causal_model, num_choices)

    return {
        "same_symbol_different_position": CounterfactualDataset.from_sampler(
            dataset_size, same_symbol_different_position_factory
        ),
        "different_symbol_different_position": CounterfactualDataset.from_sampler(
            dataset_size, different_symbol_different_position_factory
        ),
        "different_symbol_same_position": CounterfactualDataset.from_sampler(
            dataset_size, different_symbol_same_position_factory
        ),
        "random_counterfactual": CounterfactualDataset.from_sampler(
            dataset_size, random_counterfactual_factory
        ),
    }


# # Loading in a Language Model

# Definition token positions of interest

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


def create_token_positions(pipeline, causal_model):
    """Create TokenPosition objects for the experiment."""
    return [
        TokenPosition(
            lambda x: get_correct_symbol_index(x, pipeline, causal_model),
            pipeline,
            id="correct_symbol",
        ),
        TokenPosition(
            lambda x: [get_correct_symbol_index(x, pipeline, causal_model)[0] + 1],
            pipeline,
            id="correct_symbol_period",
        ),
        TokenPosition(
            lambda x: get_last_token_index(x, pipeline), pipeline, id="last_token"
        ),
    ]


def get_dtype(dtype_str: str):
    """Convert dtype string to torch dtype."""
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "float32":
        return torch.float32
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        return torch.float32


# Checker function
def checker(neural_output, causal_output):
    return neural_output in causal_output


@hydra.main(version_base=None, config_path="config", config_name="mcqa_tutorial")
def main(cfg: DictConfig) -> None:
    """Main function to run the MCQA tutorial experiment."""

    print("Starting MCQA Tutorial with Hydra configuration...")

    # Create causal model
    causal_model = create_causal_model(cfg.data.num_choices)

    # Show example
    example = sample_answerable_question(causal_model, cfg.data.num_choices)
    full_setting = causal_model.run_forward(example)
    print("Example input:")
    for k, v in example.items():
        print(f"{k}: {v}")
    print("\nFull setting:")
    for k, v in full_setting.items():
        print(f"{k}: {v}")

    # Create counterfactual datasets
    if cfg.data.use_hf:
        # Use the provided datasets from tasks.simple_MCQA
        counterfactual_datasets = get_counterfactual_datasets(
            hf=cfg.data.use_hf,
            size=cfg.data.dataset_size,
            load_private_data=cfg.data.load_private_data,
        )
        causal_model = get_causal_model()
    else:
        # Use our custom datasets
        counterfactual_datasets = create_counterfactual_datasets(
            causal_model, cfg.data.num_choices, cfg.data.dataset_size
        )

    # Show dataset examples
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

    # Setup device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = LMPipeline(
        cfg.model.name,
        max_new_tokens=cfg.model.max_new_tokens,
        device=device,
        dtype=get_dtype(cfg.model.dtype),
    )
    pipeline.tokenizer.padding_side = "left"
    print("DEVICE:", pipeline.model.device)

    # Get a sample input and check model's prediction
    sampled_example = next(iter(counterfactual_datasets.values()))[0]
    print("INPUT:", sampled_example["input"])
    print(
        "EXPECTED OUTPUT:",
        causal_model.run_forward(sampled_example["input"])["raw_output"],
    )
    print(
        "MODEL PREDICTION:", pipeline.dump(pipeline.generate(sampled_example["input"]))
    )

    # Create token positions
    if cfg.data.use_hf:
        token_positions = get_token_positions(pipeline, causal_model)
    else:
        token_positions = create_token_positions(pipeline, causal_model)

    # DAS config
    config = {
        "batch_size": cfg.training.batch_size,
        "evaluation_batch_size": cfg.training.evaluation_batch_size,
        "training_epoch": cfg.training.training_epoch,
        "n_features": cfg.training.n_features,
        "init_lr": cfg.training.init_lr,
        "temperature_schedule": tuple(cfg.training.temperature_schedule),
        "regularization_coefficient": cfg.training.regularization_coefficient,
        "mask_intervention_kwargs": dict(cfg.training.mask_intervention),
    }

    # Determine layers to use
    if cfg.experiment.use_all_layers:
        layers = list(range(pipeline.get_num_layers()))
    else:
        layers = (
            cfg.experiment.specific_layers if cfg.experiment.specific_layers else [1]
        )

    # PatchResidualStream experiment
    experiment = PatchResidualStream(
        pipeline=pipeline,
        causal_model=causal_model,
        layers=layers,
        token_positions=token_positions,
        checker=checker,
        config=config,
    )

    # Train interventions
    experiment.train_interventions(
        counterfactual_datasets,
        cfg.experiment.target_variables,
        method=cfg.experiment.method,
        verbose=cfg.experiment.verbose,
    )

    # Perform interventions
    raw_results = experiment.perform_interventions(
        counterfactual_datasets,
        verbose=cfg.experiment.verbose,
        target_variables_list=[cfg.experiment.target_variables],
        save_dir=cfg.experiment.save_dir,
    )

    # Plot results
    print(f"\nHeatmaps for '{cfg.experiment.target_variables[0]}' variable:")
    experiment.plot_heatmaps(
        raw_results, cfg.experiment.target_variables, save_path=cfg.experiment.save_dir
    )


if __name__ == "__main__":
    main()
