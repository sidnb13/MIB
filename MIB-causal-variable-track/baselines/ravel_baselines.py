import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tasks.RAVEL.ravel import get_token_positions, get_counterfactual_datasets, get_causal_model
from experiments.aggregate_experiments import residual_stream_baselines
from neural.pipeline import LMPipeline
from experiments.filter_experiment import FilterExperiment
from causal.counterfactual_dataset import CounterfactualDataset
import torch
import gc
import os
import re
import random
from copy import deepcopy

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAVEL experiments with optional flags.")
    parser.add_argument("--skip_gemma", action="store_true", help="Skip running experiments for Gemma model.")
    parser.add_argument("--skip_llama", action="store_true", help="Skip running experiments for Llama model.")
    parser.add_argument("--use_gpu1", action="store_true", help="Use GPU1 instead of GPU0 if available.")
    parser.add_argument("--methods", nargs="+", 
                        default=["full_vector", "DAS", "DBM+SVD", "DBM+PCA", "DBM", "DBM+SAE"], 
                        help="List of methods to run")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=512, help="Batch size for evaluation")
    parser.add_argument("--results_dir", type=str, default="results_ravel", help="Directory to save results")
    parser.add_argument("--model_dir", type=str, default="ravel_models", help="Directory to save trained models")
    parser.add_argument("--quick_test", action="store_true", help="Run quick test with reduced dataset size and layers")
    args = parser.parse_args()

    # Clear memory before starting
    gc.collect()
    torch.cuda.empty_cache()

    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.use_gpu1 and torch.cuda.is_available():
        device = "cuda:1"
    
    # Check function for evaluating model outputs
    def checker(output_text, expected):
        if output_text is None:
            return False

        output_clean = re.sub(r'[^\w\s]+', '', output_text.lower()).strip()
        expected_list = [e.strip().lower() for e in expected.split(',')]

        if any(part in output_clean for part in expected_list):
            return True
        
        # Edge cases
        if re.search(r'united states|united kingdom|czech republic', expected, re.IGNORECASE):
            raw_expected = expected.strip().lower().replace('the ', '')
            raw_output = output_text.strip().lower().replace('the ', '')
            if raw_output.startswith(raw_expected) or raw_output.startswith('england') or raw_output == "us":
                return True
        if re.search(r'south korea', expected, re.IGNORECASE):
            if output_clean.startswith('korea') or output_clean.startswith('south korea'):
                return True
        if re.search(r'persian|farsi', expected, re.IGNORECASE):
            if output_clean.startswith('persian') or output_clean.startswith('farsi'):
                return True
        if re.search(r'oceania', expected, re.IGNORECASE):
            if output_clean.startswith('australia'):
                return True
        if re.search(r'north america', expected, re.IGNORECASE):
            if 'north america' in output_clean or output_clean == 'na' or output_clean.startswith('america'):
                return True
        if re.search(r'mandarin|chinese', expected, re.IGNORECASE):
            if 'chinese' in output_clean or 'mandarin' in output_clean:
                return True

        return False

    # Function to clear memory between experiments
    def clear_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_filtered_indices(dataset, variables_list, target_variable):
        """
        Return a list of row indices to keep, where we keep total_size * ratio rows for each attribute.
        """
        attr_to_indices = {attr: [] for attr in variables_list}
        for i in range(len(dataset)):
            example = dataset[i]
            # Access the input dict to get the queried_attribute
            if isinstance(example["input"], dict):
                attr = example["input"].get("queried_attribute")

            
            if attr in attr_to_indices:
                attr_to_indices[attr].append(i)
        
        half = len(attr_to_indices[target_variable])
        for attr in variables_list:
            random.shuffle(attr_to_indices[attr])

        final_indices = []
        for attr in variables_list:
            if attr == target_variable: 
                final_indices.extend(attr_to_indices[attr][:half])
            else:
                final_indices.extend(attr_to_indices[attr][:int(half//2)])
                
        random.shuffle(final_indices)
        return final_indices

    def filter_dataset_by_attribute(dataset, variables_list, target_variable):
        """
        Filter a CounterfactualDataset by attribute, returning a new filtered dataset.
        """
        # Get the indices to filter
        final_indices = get_filtered_indices(dataset.dataset, variables_list=variables_list, target_variable=target_variable)
        
        # Create new filtered dataset
        filtered_hf_dataset = dataset.dataset.select(final_indices)
        
        # Return a new CounterfactualDataset with the filtered data
        return CounterfactualDataset(dataset=filtered_hf_dataset, id=dataset.id)

    # Get counterfactual datasets and causal model once
    dataset_size = 10 if args.quick_test else 10000
    counterfactual_datasets = get_counterfactual_datasets(hf=True, size=dataset_size)
    causal_model = get_causal_model()

    # Print available datasets
    print("Available datasets:", counterfactual_datasets.keys())
    
    # Set up models to test
    models = []
    if not args.skip_gemma:
        models.append("google/gemma-2-2b")
    if not args.skip_llama:
        models.append("meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    # Make sure results and model directories exist
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    for model_name in models:
        print(f"\n===== Testing model: {model_name} =====")
        
        # Set up LM Pipeline
        pipeline = LMPipeline(model_name, max_new_tokens=2, device=device, dtype=torch.float16)
        pipeline.tokenizer.padding_side = "left"
        print("DEVICE:", pipeline.model.device)
        
        # Model-specific batch size adjustments
        if "gemma" in model_name:
            model_batch_size = args.batch_size
            model_eval_batch_size = args.eval_batch_size
        else:
            # Llama typically needs smaller batch sizes
            model_batch_size =  32
            model_eval_batch_size = 256
        
        # Get a sample input and check model's prediction
        sampled_example = next(iter(counterfactual_datasets.values()))[0]
        print("INPUT:", sampled_example["input"])
        print("EXPECTED OUTPUT:", causal_model.run_forward(sampled_example["input"])["raw_output"])
        print("MODEL PREDICTION:", pipeline.dump(pipeline.generate(sampled_example["input"])))
        
        # Filter the datasets based on model performance
        print("\nFiltering datasets based on model performance...")
        exp = FilterExperiment(pipeline, causal_model, checker)
        filtered_datasets = exp.filter(counterfactual_datasets, verbose=True, batch_size=model_eval_batch_size)
        
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
        start = 0
        end = 1 if args.quick_test else pipeline.get_num_layers()
        
        if "gemma" in model_name:
            config = {
                "batch_size": 128, 
                "evaluation_batch_size": 512, 
                "training_epoch": 1, 
                "n_features": 288, 
                "regularization_coefficient": 0.0, 
                "output_scores": False
            }
        else:
            config = {
                "batch_size": 48, 
                "evaluation_batch_size": 256, 
                "training_epoch": 1, 
                "n_features": 512, 
                "regularization_coefficient": 0.0, 
                "output_scores": False
            }
        
        all_attributes = ["Continent", "Country", "Language"]

        # Set which variables to localize
        target_variables = [ "Country", "Language", "Continent"]
        
        # Set which counterfactuals to use
        names = ["attribute", "wikipedia"]

        # Run experiments for each target variable
        for variable in target_variables:
            print(f"\nRunning experiments for target variable: {variable}")
            
            # Create deep copies of filtered datasets for attribute-specific filtering
            temp_datasets = deepcopy(filtered_datasets)
            
            # Prepare train and test data dictionaries with attribute-specific filtering
            train_data = {}
            test_data = {}
            
            for name in names:
                # Process training data
                if name + "_train" in temp_datasets:
                    print(f"Original {name}_train size: {len(temp_datasets[name + '_train'])}")
                    filtered_train = filter_dataset_by_attribute(
                        temp_datasets[name + "_train"], 
                        variables_list=all_attributes, 
                        target_variable=variable
                    )
                    print(f"Filtered {name}_train size: {len(filtered_train)}")
                    train_data[name + "_train"] = filtered_train
                
                # Process test data
                if name + "_test" in temp_datasets:
                    print(f"Original {name}_test size: {len(temp_datasets[name + '_test'])}")
                    filtered_test = filter_dataset_by_attribute(
                        temp_datasets[name + "_test"], 
                        variables_list=all_attributes, 
                        target_variable=variable
                    )
                    print(f"Filtered {name}_test size: {len(filtered_test)}")
                    test_data[name + "_test"] = filtered_test
                
                # Process private test data
                if name + "_testprivate" in temp_datasets:
                    print(f"Original {name}_testprivate size: {len(temp_datasets[name + '_testprivate'])}")
                    filtered_test_private = filter_dataset_by_attribute(
                        temp_datasets[name + "_testprivate"], 
                        variables_list=all_attributes, 
                        target_variable=variable
                    )
                    print(f"Filtered {name}_testprivate size: {len(filtered_test_private)}")
                    test_data[name + "_testprivate"] = filtered_test_private
            
            # Run the baseline experiments
            residual_stream_baselines(
                pipeline=pipeline,
                task=causal_model,
                token_positions=token_positions,
                train_data=train_data,
                test_data=test_data,
                config=config,
                target_variables=[variable],
                checker=checker,
                start=start,
                end=end,
                verbose=True,
                model_dir=os.path.join(args.model_dir, variable),
                results_dir=args.results_dir,
                methods=args.methods
            )
            clear_memory()
        
        # Clean up pipeline to free memory before starting next model
        del pipeline
        clear_memory()
    
    print("\nAll experiments completed.")