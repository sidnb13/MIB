import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "experiments"))
sys.path.append(
    str(Path(__file__).resolve().parent.parent / "CausalAbstraction")
)

import gc
import os

import hydra
import torch
from experiments.aggregate_experiments import residual_stream_baselines
from experiments.filter_experiment import FilterExperiment
from neural.pipeline import LMPipeline
from omegaconf import DictConfig
from tasks.two_digit_addition_task.arithmetic import (
    get_causal_model,
    get_counterfactual_datasets,
    get_token_positions,
)


def get_dtype(dtype_str: str):
    """Convert string dtype to torch dtype."""
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "float32":
        return torch.float32
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        return torch.float16  # default


def clear_memory():
    """Function to clear memory between experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def checker(output_text, expected):
    """Check function for evaluating model outputs."""
    # Clean the output by extracting just the numbers
    import re

    numbers_in_output = re.findall(r"\d+", output_text)
    if not numbers_in_output:
        return False

    # Get the first number found
    first_number = numbers_in_output[0]

    # Handle the case where expected has leading zero and output doesn't
    if expected[0] == "0":
        expected_no_leading_zero = expected[1:]
        return first_number == expected_no_leading_zero or first_number == expected
    return first_number == expected


@hydra.main(
    version_base=None, config_path="../config", config_name="arithmetic_baseline"
)
def main(cfg: DictConfig) -> None:
    """Main function to run arithmetic experiments with Hydra configuration."""

    print("Starting Arithmetic Experiments with Hydra configuration...")

    # Clear memory before starting
    clear_memory()

    # Device setup
    device = cfg.model.device if torch.cuda.is_available() else "cpu"
    if cfg.model.use_gpu1 and torch.cuda.is_available():
        device = "cuda:1"

    # Get counterfactual datasets and causal model
    dataset_size = (
        cfg.data.quick_test_size if cfg.data.quick_test else cfg.data.dataset_size
    )
    counterfactual_datasets = get_counterfactual_datasets(hf=True, size=dataset_size)
    causal_model = get_causal_model()

    # Print available datasets
    print("Available datasets:", counterfactual_datasets.keys())

    # Set up models to test
    models = []
    if not cfg.model.skip_gemma:
        models.append("google/gemma-2-2b")
    if not cfg.model.skip_llama:
        models.append("meta-llama/Meta-Llama-3.1-8B-Instruct")

    for model_name in models:
        print(f"\n===== Testing model: {model_name} =====")

        # Set up LM Pipeline with appropriate max_new_tokens for each model
        if "llama" in model_name.lower():
            max_new_tokens = cfg.model.max_new_tokens.llama
        elif "gemma" in model_name.lower():
            max_new_tokens = cfg.model.max_new_tokens.gemma
        else:
            max_new_tokens = cfg.model.max_new_tokens.default

        pipeline = LMPipeline(
            model_name,
            max_new_tokens=max_new_tokens,
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
            "MODEL PREDICTION:",
            pipeline.dump(pipeline.generate(sampled_example["input"])),
        )

        # Filter the datasets based on model performance
        print("\nFiltering datasets based on model performance...")
        exp = FilterExperiment(pipeline, causal_model, checker)
        filtered_datasets = exp.filter(
            counterfactual_datasets,
            verbose=cfg.experiment.verbose,
            batch_size=cfg.training.evaluation_batch_size,
        )

        # Get token positions for intervention
        token_positions = get_token_positions(pipeline, causal_model)

        # Display token highlighting for a sample
        print("\nToken positions highlighted in samples:")
        for dataset in filtered_datasets.values():
            for token_position in token_positions:
                example = dataset[0]
                print(token_position.highlight_selected_token(example["input"]))
                break
            break

        # Clear memory before running experiments
        clear_memory()

        # Setup experiment configuration
        start = cfg.experiment.layers.start
        if cfg.data.quick_test:
            end = cfg.experiment.layers.quick_test_end
        else:
            end = (
                cfg.experiment.layers.end
                if cfg.experiment.layers.end is not None
                else pipeline.get_num_layers()
            )

        # Base configuration
        config = {
            "batch_size": cfg.training.batch_size,
            "evaluation_batch_size": cfg.training.evaluation_batch_size,
            "training_epoch": cfg.training.training_epoch,
            "n_features": cfg.training.n_features,
            "regularization_coefficient": cfg.training.regularization_coefficient,
            "output_scores": cfg.training.output_scores,
        }

        # Adjust batch size for specific models
        if "llama" in model_name.lower() and "llama" in cfg.training.model_specific:
            config["batch_size"] = cfg.training.model_specific.llama.batch_size
            config["evaluation_batch_size"] = (
                cfg.training.model_specific.llama.evaluation_batch_size
            )

        # Make sure results and model directories exist
        if not os.path.exists(cfg.experiment.results_dir):
            os.makedirs(cfg.experiment.results_dir)

        if not os.path.exists(cfg.experiment.model_dir):
            os.makedirs(cfg.experiment.model_dir)

        # Run experiments for target variables
        print(
            f"\nRunning experiments for target variables: {cfg.data.target_variables}"
        )

        # Prepare train and test data dictionaries
        train_data = {}
        test_data = {}

        for name in cfg.data.names:
            if name + "_train" in filtered_datasets:
                train_data[name + "_train"] = filtered_datasets[name + "_train"]
            if name + "_test" in filtered_datasets:
                test_data[name + "_test"] = filtered_datasets[name + "_test"]
            if name + "_testprivate" in filtered_datasets:
                test_data[name + "_testprivate"] = filtered_datasets[
                    name + "_testprivate"
                ]

        # Add mask intervention kwargs for DBM_NEW if it's in the methods
        if "DBM_NEW" in cfg.experiment.methods:
            config["mask_intervention_kwargs"] = dict(
                cfg.training.mask_intervention_kwargs
            )

        # Run all experiments with the same config
        residual_stream_baselines(
            pipeline=pipeline,
            task=causal_model,
            token_positions=token_positions,
            train_data=train_data,
            test_data=test_data,
            config=config,
            target_variables=cfg.data.target_variables,
            checker=checker,
            start=start,
            end=end,
            verbose=cfg.experiment.verbose,
            model_dir=os.path.join(
                cfg.experiment.model_dir, "_".join(cfg.data.target_variables)
            ),
            results_dir=cfg.experiment.results_dir,
            methods=cfg.experiment.methods,
        )

        clear_memory()

        # Clean up pipeline to free memory before starting next model
        del pipeline
        clear_memory()

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
