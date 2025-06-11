import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from tasks.IOI_task.ioi_task import get_causal_model, get_counterfactual_datasets, get_token_positions
from CausalAbstraction.experiments.aggregate_experiments import attention_head_baselines 
from CausalAbstraction.experiments.attention_head_experiment import PatchAttentionHeads
from CausalAbstraction.experiments.filter_experiment import FilterExperiment
from ioi_utils import (log_diff, clear_memory, checker, filter_checker, custom_loss, 
                       ioi_loss_and_metric_fn, setup_pipeline, get_model_config)
from itertools import combinations

import torch
import gc
import json
import os


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run IOI experiments with pre-computed linear parameters.")
    parser.add_argument("--model", type=str, required=True, choices=["gpt2", "qwen", "llama", "gemma"],
                        help="Model to use for experiments")
    parser.add_argument("--use_gpu1", action="store_true", help="Use GPU1 instead of GPU0 if available.")
    parser.add_argument("--linear_params", type=str, required=True,
                        help="Linear model coefficients. Can be either a JSON file path or a dict string like \"{'bias': 0.295, 'token_coeff': 0.63, 'position_coeff': 2.235}\"")
    parser.add_argument("--heads_list", nargs="+", type=lambda s: eval(s), 
                        default=[(7, 3), (7, 9), (8, 6), (8, 10)], 
                        help="List of (layer, head) tuples to intervene on. Example: '(7,3)' '(7,9)'")
    parser.add_argument("--methods", nargs="+", 
                        default=["full_vector", "DAS", "DBM+SVD", "DBM+PCA", "DBM"], 
                        help="List of methods to run")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training (uses model default if not specified)")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Batch size for evaluation (uses model default if not specified)")
    parser.add_argument("--results_dir", type=str, default="ioi_results", help="Directory to save results")
    parser.add_argument("--model_dir", type=str, default="ioi_models", help="Directory to save trained models")
    parser.add_argument("--skip_output_token", action="store_true", help="Skip experiments for output_token variable.")
    parser.add_argument("--skip_output_position", action="store_true", help="Skip experiments for output_position variable.")
    parser.add_argument("--run_baselines", action="store_true", help="Run baseline experiments for output_position and output_token.")
    parser.add_argument("--run_brute_force", action="store_true", help="Run brute force head subset experiments.")
    parser.add_argument("--quick_test", action="store_true", help="Run quick test with reduced dataset size and layers")
    args = parser.parse_args()

    # Clear memory before starting
    gc.collect()
    torch.cuda.empty_cache()

    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.use_gpu1 and torch.cuda.is_available():
        device = "cuda:1"

    # Load linear parameters
    print(f"Loading linear parameters from: {args.linear_params}")
    try:
        if os.path.isfile(str(args.linear_params)):
            with open(args.linear_params, 'r') as f:
                all_coeffs = json.load(f)
        else:
            # Try to evaluate as dict string
            coeffs = eval(args.linear_params)
            # Convert single dict to the expected format
            all_coeffs = {args.model: coeffs}
    except Exception as e:
        raise ValueError(f"Failed to parse linear_params: {e}")

    # Find the coefficients for this model
    if args.model in all_coeffs:
        coeffs = all_coeffs[args.model]
    elif "default" in all_coeffs:
        coeffs = all_coeffs["default"]
    else:
        # Use the first available coefficients
        coeffs = next(iter(all_coeffs.values()))
    
    # Validate required keys
    required_keys = ['bias', 'token_coeff', 'position_coeff']
    for key in required_keys:
        if key not in coeffs:
            raise ValueError(f"Missing required key '{key}' in linear_coeffs for model {args.model}")
    
    intercept = coeffs['bias']
    token_coef = coeffs['token_coeff']
    position_coef = coeffs['position_coeff']
    
    print(f"Using coefficients for {args.model}:")
    print(f"  bias: {intercept}")
    print(f"  token_coeff: {token_coef}")
    print(f"  position_coeff: {position_coef}")
    
    # Update the causal model's mechanism
    causal_model = get_causal_model({"bias": intercept, "token_coeff": token_coef, "position_coeff": position_coef})
    
    # Limit heads_list for quick test
    if args.quick_test and len(args.heads_list) > 1:
        args.heads_list = args.heads_list[:2]  # Use only the first two heads for quick test
        print(f"Quick test mode: limiting to heads {args.heads_list}")
    
    # Get counterfactual datasets
    dataset_size = 10 if args.quick_test else None
    counterfactual_datasets = get_counterfactual_datasets(hf=True, size=dataset_size)

    # Print dataset info
    print("Available datasets:", counterfactual_datasets.keys())
    
    # Get a sample to display
    sample_dataset = next(iter(counterfactual_datasets.values()))
    if len(sample_dataset) > 0:
        sample = sample_dataset[0]
        print("Sample input:", sample["input"])

    print(f"\n===== Testing model: {args.model} =====")
    
    # Set up pipeline
    pipeline, default_batch_size = setup_pipeline(args.model, device, args.eval_batch_size)
    batch_size = args.batch_size if args.batch_size else default_batch_size
    eval_batch_size = args.eval_batch_size if args.eval_batch_size else default_batch_size
    
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
    filtered_datasets = exp.filter(counterfactual_datasets, verbose=True, batch_size=eval_batch_size)
    
    # Get token positions
    token_positions = get_token_positions(pipeline, causal_model)
    
    # Clear memory before running experiments
    clear_memory()

    # Setup experiment configuration
    
    config = {
        "evaluation_batch_size": eval_batch_size,
        "batch_size": batch_size, 
        "training_epoch": 2, 
        "check_raw": True,
        "n_features": 32, 
        "regularization_coefficient": 0.0, 
        "output_scores": True, 
        "shuffle": True, 
        "temperature_schedule": (1.0, 0.01), 
        "init_lr": 1.0,
        "loss_and_metric_fn": lambda pipeline, intervenable_model, batch, model_units_list: 
            ioi_loss_and_metric_fn(pipeline, intervenable_model, batch, model_units_list),
    }
    
    # Setup counterfactual names
    counterfactuals = ["s1_io_flip", "s2_io_flip", "s1_ioi_flip_s2_ioi_flip"]
    train_data = {}
    test_data = {}
    
    for counterfactual in counterfactuals:
        if counterfactual + "_train" in filtered_datasets:
            train_data[counterfactual + "_train"] = filtered_datasets[counterfactual + "_train"]
        if counterfactual + "_test" in filtered_datasets:
            test_data[counterfactual + "_test"] = filtered_datasets[counterfactual + "_test"]
        if counterfactual + "_testprivate" in filtered_datasets:
            test_data[counterfactual + "_testprivate"] = filtered_datasets[counterfactual + "_testprivate"]
    
    verbose = True
    
    if args.run_baselines:
        # Make sure results and model directories exist
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        
        # Run experiments for output_position
        if not args.skip_output_position:
            print(f"\nRunning experiments for target variable: output_position")
            
            attention_head_baselines(
                pipeline=pipeline, 
                task=causal_model, 
                token_positions=token_positions, 
                train_data=train_data, 
                test_data=test_data, 
                config=config, 
                target_variables=["output_position"], 
                checker=lambda logits, params: checker(logits, params, pipeline), 
                verbose=verbose, 
                results_dir=args.results_dir,
                heads_list=args.heads_list,
                skip=[method for method in ["full_vector", "DAS", "DBM+SVD", "DBM+PCA", "DBM", "DBM+SAE"] if method not in args.methods]
            )
            clear_memory()
        
        # Run experiments for output_token
        if not args.skip_output_token:
            print(f"\nRunning experiments for target variable: output_token")
            
            attention_head_baselines(
                pipeline=pipeline, 
                task=causal_model, 
                token_positions=token_positions, 
                train_data=train_data, 
                test_data=test_data, 
                config=config, 
                target_variables=["output_token"], 
                checker=lambda logits, params: checker(logits, params, pipeline), 
                verbose=verbose, 
                results_dir=args.results_dir,
                heads_list=args.heads_list,
                skip=[method for method in ["full_vector", "DAS", "DBM+SVD", "DBM+PCA", "DBM", "DBM+SAE"] if method not in args.methods]
            )
            clear_memory()
    
    if args.run_brute_force:
        # Add this code to the end of ioi_baselines.py script
        print("\n" + "="*60)
        print("RUNNING HEAD SUBSET EXPERIMENTS")
        print("="*60)
        
        # Generate all non-empty proper subsets of heads_list (excluding the full set)
        all_subsets = []
        for r in range(1, len(args.heads_list)):  # 1 to len-1, excluding full set
            all_subsets.extend(combinations(args.heads_list, r))
        
        print(f"Testing {len(all_subsets)} head subsets...")
        
        # Results storage
        subset_results = {}
        
        for i, head_subset in enumerate(all_subsets):
            head_subset = list(head_subset)  # Convert tuple to list
            subset_name = "_".join([f"L{layer}H{head}" for layer, head in head_subset])
            
            print(f"\n[{i+1}/{len(all_subsets)}] Testing heads: {head_subset}")
            
            try:
                # Create experiment with current head subset
                experiment = PatchAttentionHeads(
                    pipeline=pipeline,
                    causal_model=causal_model,
                    layer_head_list=head_subset,
                    token_positions=token_positions,
                    checker=lambda logits, params: checker(logits, params, pipeline),
                    config={**config, "method_name": "full_vector"}
                )
                
                # Test both target variables if not skipped
                subset_results[subset_name] = {}
                
                if not args.skip_output_position:
                    print(f"  Testing output_position...")
                    position_results = experiment.perform_interventions(
                        test_data,
                        verbose=False,
                        target_variables_list=[["output_position"]],
                        save_dir=None  # Don't save individual results
                    )
                    
                    # Group scores by test vs private test
                    subset_results[subset_name]["output_position"] = {}
                    test_scores = []
                    private_scores = []
                    
                    for dataset_name in position_results["dataset"]:
                        dataset_scores = []
                        for unit_data in position_results["dataset"][dataset_name]["model_unit"].values():
                            if "output_position" in unit_data and "average_score" in unit_data["output_position"]:
                                dataset_scores.append(unit_data["output_position"]["average_score"])
                        
                        if dataset_scores:
                            avg_score = sum(dataset_scores) / len(dataset_scores)
                            if "testprivate" in dataset_name:
                                private_scores.append(avg_score)
                            else:
                                test_scores.append(avg_score)
                    
                    if test_scores:
                        subset_results[subset_name]["output_position"]["test"] = sum(test_scores) / len(test_scores)
                        print(f"    Position test: {subset_results[subset_name]['output_position']['test']:.4f}")
                    
                    if private_scores:
                        subset_results[subset_name]["output_position"]["private"] = sum(private_scores) / len(private_scores)
                        print(f"    Position private: {subset_results[subset_name]['output_position']['private']:.4f}")
                
                if not args.skip_output_token:
                    print(f"  Testing output_token...")
                    token_results = experiment.perform_interventions(
                        test_data,
                        verbose=False,
                        target_variables_list=[["output_token"]],
                        save_dir=None  # Don't save individual results
                    )
                    
                    # Group scores by test vs private test
                    subset_results[subset_name]["output_token"] = {}
                    test_scores = []
                    private_scores = []
                    
                    for dataset_name in token_results["dataset"]:
                        dataset_scores = []
                        for unit_data in token_results["dataset"][dataset_name]["model_unit"].values():
                            if "output_token" in unit_data and "average_score" in unit_data["output_token"]:
                                dataset_scores.append(unit_data["output_token"]["average_score"])
                        
                        if dataset_scores:
                            avg_score = sum(dataset_scores) / len(dataset_scores)
                            if "testprivate" in dataset_name:
                                private_scores.append(avg_score)
                            else:
                                test_scores.append(avg_score)
                    
                    if test_scores:
                        subset_results[subset_name]["output_token"]["test"] = sum(test_scores) / len(test_scores)
                        print(f"    Token test: {subset_results[subset_name]['output_token']['test']:.4f}")
                    
                    if private_scores:
                        subset_results[subset_name]["output_token"]["private"] = sum(private_scores) / len(private_scores)
                        print(f"    Token private: {subset_results[subset_name]['output_token']['private']:.4f}")
                
                # Clean up experiment
                del experiment
                clear_memory()
                
            except Exception as e:
                print(f"  Error with subset {head_subset}: {e}")
                subset_results[subset_name] = {"error": str(e)}
                clear_memory()
                continue
        
        # Save results to file
        results_filename = f"head_subset_results_{args.model}.json"
        results_path = os.path.join(args.results_dir, results_filename)
        
        with open(results_path, 'w') as f:
            json.dump(subset_results, f, indent=2)
        
        print(f"\nHead subset results saved to: {results_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("HEAD SUBSET EXPERIMENT SUMMARY")
        print("="*60)
        
        for subset_name, scores in subset_results.items():
            if "error" not in scores:
                print(f"{subset_name}:")
                if "output_position" in scores:
                    if "test" in scores["output_position"]:
                        print(f"  Position test: {scores['output_position']['test']:.4f}")
                    if "private" in scores["output_position"]:
                        print(f"  Position private: {scores['output_position']['private']:.4f}")
                if "output_token" in scores:
                    if "test" in scores["output_token"]:
                        print(f"  Token test: {scores['output_token']['test']:.4f}")
                    if "private" in scores["output_token"]:
                        print(f"  Token private: {scores['output_token']['private']:.4f}")
            else:
                print(f"{subset_name}: ERROR - {scores['error']}")
        
        print("\nHead subset experiments completed!")