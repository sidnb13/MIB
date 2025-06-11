import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from tasks.IOI_task.ioi_task import get_causal_model, get_counterfactual_datasets, get_token_positions
from CausalAbstraction.experiments.filter_experiment import FilterExperiment
from CausalAbstraction.experiments.attention_head_experiment import PatchAttentionHeads
from ioi_utils import log_diff, clear_memory, checker, filter_checker, setup_pipeline
import torch
import gc
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import json

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute linear model parameters for IOI experiments.")
    parser.add_argument("--model", type=str, required=True, choices=["gpt2", "qwen", "llama", "gemma"],
                        help="Model to use for parameter computation")
    parser.add_argument("--use_gpu1", action="store_true", help="Use GPU1 instead of GPU0 if available.")
    parser.add_argument("--heads_list", nargs="+", type=lambda s: eval(s), 
                        default=[(7, 3), (7, 9), (8, 6), (8, 10)], 
                        help="List of (layer, head) tuples to intervene on. Example: '(7,3)' '(7,9)'")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Batch size for evaluation (uses model default if not specified)")
    parser.add_argument("--output_file", type=str, default="ioi_linear_params.json", help="Output file for linear parameters")
    parser.add_argument("--quick_test", action="store_true", help="Run quick test with reduced dataset size")
    args = parser.parse_args()

    # Clear memory before starting
    gc.collect()
    torch.cuda.empty_cache()

    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.use_gpu1 and torch.cuda.is_available():
        device = "cuda:1"

    # Get causal model and counterfactual datasets
    causal_model = get_causal_model({"bias": 0.0, "token_coeff": 0.0, "position_coeff": 0.0})
    dataset_size = 10 if args.quick_test else None
    counterfactual_datasets = get_counterfactual_datasets(hf=True, size=dataset_size)

    # Print dataset info
    print("Available datasets:", counterfactual_datasets.keys())
    
    # Get a sample to display
    sample_dataset = next(iter(counterfactual_datasets.values()))
    if len(sample_dataset) > 0:
        sample = sample_dataset[0]
        print("Sample input:", sample["input"])

    print(f"\n===== Computing parameters for model: {args.model} =====")
    
    # Set up pipeline
    pipeline, batch_size = setup_pipeline(args.model, device, args.eval_batch_size)
    print("DEVICE:", pipeline.model.device)
    
    # Test model on a sample
    if len(sample_dataset) > 0:
        sample = sample_dataset[0]
        print("INPUT:", sample["input"]["raw_input"])
        expected = causal_model.run_forward(sample["input"])["raw_output"]
        print("EXPECTED OUTPUT:", expected)
        print("MODEL PREDICTION:", pipeline.dump(pipeline.generate(sample["input"]["raw_input"])))
    
    # Filter the datasets
    print("\nFiltering datasets based on model performance...")
    exp = FilterExperiment(pipeline, causal_model, filter_checker)
    filtered_datasets = exp.filter(counterfactual_datasets, verbose=True, batch_size=batch_size)
    
    # Get token positions
    token_positions = get_token_positions(pipeline, causal_model)
    
    # Limit heads_list for quick test
    if args.quick_test and len(args.heads_list) > 1:
        args.heads_list = args.heads_list[:1]  # Use only the first head for quick test
        print(f"Quick test mode: limiting to heads {args.heads_list}")
    
    print("\nFitting linear model for logit differences...")
    
    # Set up for return_scores
    pipeline.return_scores = True
    
    # Collect data for linear regression
    data_to_X = {
        "same_train": {"position": 1, "token": 1}, 
        "s1_io_flip_train": {"position": -1, "token": 1},
        "s2_io_flip_train": {"position": -1, "token": -1},
        "s1_ioi_flip_s2_ioi_flip_train": {"position": 1, "token": -1}
    }
    
    # Limit datasets for quick test
    if args.quick_test:
        # Use only first two datasets for quick test
        data_to_X = dict(list(data_to_X.items())[:2])
    X, y = [], []
    
    for counterfactual_name in data_to_X:
        if counterfactual_name not in filtered_datasets:
            print(f"Warning: {counterfactual_name} not found in filtered datasets, skipping...")
            continue
            
        experiment = PatchAttentionHeads(
            pipeline=pipeline, 
            causal_model=causal_model, 
            layer_head_list=args.heads_list,
            token_positions=token_positions, 
            checker=lambda logits, params: checker(logits, params, pipeline), 
            config={"evaluation_batch_size": batch_size, "output_scores": True, "check_raw":True}
        )
        
        raw_results = experiment.perform_interventions(
            {counterfactual_name: filtered_datasets[counterfactual_name]}, 
            target_variables_list=[["output_token"]],
            verbose=False
        )
        
        raw_outputs = None
        losses, labels, counterfactual_y = [], [], []
        
        for v in raw_results["dataset"][counterfactual_name].values():
            for v2 in v.values():
                raw_outputs = v2["raw_outputs"][0]
        
        for raw_logits, input_data in zip(raw_outputs, filtered_datasets[counterfactual_name]):
            actual_diff = log_diff(raw_logits, causal_model.run_forward(input_data["input"]), pipeline)
            high_level_output = causal_model.run_interchange(
                input_data["input"], 
                {"output_token": input_data["counterfactual_inputs"][0], 
                 "output_position": input_data["counterfactual_inputs"][0]}
            )
            loss = checker(raw_logits, high_level_output, pipeline)
            label = high_level_output["logit_diff"]
            
            y.append(actual_diff)
            counterfactual_y.append(actual_diff)
            X.append((data_to_X[counterfactual_name]["position"], data_to_X[counterfactual_name]["token"]))
            losses.append(loss)
            labels.append(label)
        
        # Compute and print the average y for the current counterfactual
        avg_y = sum(counterfactual_y) / len(counterfactual_y) if counterfactual_y else 0
        print(f"Average y for counterfactual '{counterfactual_name}': {avg_y}")
        print(f"Average label for counterfactual '{counterfactual_name}': {sum(labels) / len(labels)}")
        print(f"Average loss for counterfactual '{counterfactual_name}': {sum(losses) / len(losses)}")
    
    # Fit linear model
    model = LinearRegression()
    X = torch.tensor(X)
    y = torch.tensor(y)
    model.fit(X, y)
    
    # Print the coefficients
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print("Score:", model.score(X, y))
    
    intercept = float(model.intercept_)
    position_coef = float(model.coef_[0])
    token_coef = float(model.coef_[1])
    
    # Store results
    from ioi_utils import get_model_config
    model_config = get_model_config(args.model)
    model_path = model_config["model_path"]
    
    results = {
        args.model: {
            "bias": intercept,
            "position_coeff": position_coef,
            "token_coeff": token_coef,
            "score": float(model.score(X, y)),
            "model_name": model_path
        }
    }
    
    print(f"Linear parameters for {args.model}:")
    print(f"  bias: {intercept}")
    print(f"  position_coeff: {position_coef}")
    print(f"  token_coeff: {token_coef}")
    print(f"  R² score: {model.score(X, y)}")
    
    # Save results to JSON file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nLinear parameters saved to {args.output_file}")
    print("Parameter computation completed.")