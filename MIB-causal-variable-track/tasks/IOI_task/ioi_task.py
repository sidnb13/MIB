import json, random, os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from CausalAbstraction.causal.causal_model import CausalModel, CounterfactualDataset
from CausalAbstraction.neural.LM_units import TokenPosition, get_last_token_index
import copy

from copy import deepcopy
from tasks.hf_dataloader import load_hf_dataset
import re

def get_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def get_causal_model(parameters):
    """
    Create and return the causal model for IOI task.
    """
    # Load data files

    pos_coeff = parameters["position_coeff"]
    token_coeff = parameters["token_coeff"]
    bias = parameters["bias"]
    
    variables = ["raw_input", "output_position", "output_token", "name_A", "name_B", "name_C", 
                 "logit_diff", "raw_output"]

    values = {
        "raw_input": [""],  # Placeholder
        "output_position": [0, 1],
        "output_token": [""],
        "name_A": [""],
        "name_B": [""],
        "name_C": [""],
        "logit_diff": [0.0],  # Placeholder for float values
        "raw_output": [""]  # Placeholder
    }
    
    parents = {
        "raw_input":[],
        "name_A": [],
        "name_B": [],
        "name_C": [],
        "output_token": ["name_A", "name_B", "name_C"],
        "output_position": ["name_A", "name_B", "name_C"],
        "logit_diff": ["name_A", "name_B", "name_C", "output_token", "output_position"],
        "raw_output": ["output_token"]
    }

    def get_name_A():
        return ""
    
    def get_name_B():
        return ""
    
    def get_name_C():
        return ""
    
    def get_output_position(name_A, name_B, name_C):
        if name_C == name_A:
            return 1
        elif name_C == name_B:
            return 0
        else:
            return "Error"
    
    def get_output_token(name_A, name_B, name_C):
        if name_C == name_A:
            return name_B
        elif name_C == name_B:
            return name_A
        else:
            return "Error"

    def get_logit_diff(name_A, name_B, name_C, output_token, output_position):
        token_signal = None 
        if (name_C == name_A and output_token == name_B) or (name_C == name_B and output_token == name_A):
            token_signal = 1
        elif (name_C == name_A and output_token == name_A) or (name_C == name_B and output_token == name_B):
            token_signal = -1

        position_signal = None 
        if (name_C == name_A and output_position == 1) or (name_C == name_B and output_position == 0):
            position_signal = 1
        elif (name_C == name_A and output_position == 0) or (name_C == name_B and output_position == 1):
            position_signal = -1

        return bias + token_coeff * token_signal + pos_coeff * position_signal
    
    def get_raw_output(output_token):
        """Generate the raw output (just the output token)."""
        return output_token

    mechanisms = {
        "raw_input": lambda: "",
        "name_A": get_name_A,
        "name_B": get_name_B,
        "name_C": get_name_C,
        "output_token": get_output_token,
        "output_position": get_output_position,
        "logit_diff": get_logit_diff,
        "raw_output": get_raw_output
    }

    return CausalModel(variables, values, parents, mechanisms, id="ioi")

def parse_ioi_example(input):
    templates_path = os.path.join(Path(__file__).resolve().parent.parent, os.path.join("IOI_task", 'templates.json'))
    TEMPLATES = get_data(templates_path)
    # Helper to convert template into regex and track variable order
    def extract_vars(prompt):
        prompt = ' '.join(prompt.split())  # Normalize whitespace

        def template_to_regex(template):
            pattern = re.escape(template)
            var_counts = {}
            
            # Match all {var} placeholders in order
            all_vars = re.findall(r"\{(name_A|name_B|name_C|place|object)\}", template)

            for var in all_vars:
                var_counts[var] = var_counts.get(var, 0) + 1

                if var_counts[var] == 1:
                    group = f"(?P<{var}>[^,\.]+)"
                else:
                    # Avoid redefining the same named group
                    group = r"[^,\.]+"

                escaped_var = re.escape(f"{{{var}}}")
                pattern = pattern.replace(escaped_var, group, 1)  # only replace the first occurrence

            return re.compile(f"^{pattern}$")

        for template in TEMPLATES:
            regex = template_to_regex(template)
            match = regex.match(prompt)
            if match:
                return match.groupdict(), template

        print(f"Prompt '{prompt}' does not match any template.")
    output = {}
    output["raw_input"] = input["prompt"]
    if "metadata" in input:
        output["name_A"] = input["metadata"]["subject"]
        output["name_B"] = input["metadata"]["indirect_object"]
        output["name_C"] = input["metadata"]["subject"]
        output["object"] = input["metadata"]["object"] if "object" in input["metadata"] else None
        output["place"] = input["metadata"]["place"] if "place" in input["metadata"] else None
        output["template"] = input["template"]
    else:
        variables = {}
        try:
            variables, template = extract_vars(input['prompt'])
            output["name_A"] = variables["name_A"]
            output["name_B"] = variables["name_B"]
            output["name_C"] = variables["name_C"]
            output["object"] = variables["object"] if "object" in variables else None
            output["place"] = variables["place"] if "place" in variables else None
            output["template"] = template
        except Exception as e:
            print(f"Error parsing prompt: {input['prompt']} {output}")
            print(e)
            assert False
        


    return output

def get_counterfactual_datasets(hf=True, size=None):
    """
    Load and return counterfactual datasets for IOI task.
    """

    
    # Load dataset from HuggingFace with customized parsing
    datasets = {}
    for split in ["train", "test"]:
        temp = load_hf_dataset(
            dataset_path="mib-bench/ioi",
            split=split,
            parse_fn=parse_ioi_example,
            size=size,
            ignore_names=["random", "abc"]
        )
        datasets.update(temp)
    
    private = load_hf_dataset(
        dataset_path="mib-bench/ioi_private_test",
        split="test",
        parse_fn=parse_ioi_example,
        size=size,
        ignore_names=["random", "abc"]
    )
    datasets.update({k+"private":v for k,v in private.items()})
    
    # Add "same" counterfactual dataset as post-processing step
    # For each existing dataset, create a "same" version where counterfactual_inputs equals input
    same_datasets = {}
    for dataset_name, dataset in datasets.items():
        same_name = "same_" + dataset_name.split("_")[-1]  # "same_train"
        
        # Create new dataset where counterfactual_inputs = [input]
        same_data = {
            "input": [],
            "counterfactual_inputs": []
        }
        
        for example in dataset:
            same_data["input"].append(example["input"])
            same_data["counterfactual_inputs"].append([copy.deepcopy(example["input"])])
        
        same_datasets[same_name] = CounterfactualDataset.from_dict(
            same_data, 
            id=same_name
        )
    
    # Add the same datasets to the main datasets dict
    datasets.update(same_datasets)
    
    return datasets

def get_token_positions(pipeline, causal_model):
    """
    Get token positions for IOI task interventions.
    Returns all token positions.
    """
    def get_all_token_positions(input_dict, pipeline):
        """Get all token positions in the input."""
        tokens = list(range(len(pipeline.load(input_dict)['input_ids'][0])))
        return tokens
    
    return [TokenPosition(lambda x: get_all_token_positions(x, pipeline), pipeline, id="all")]