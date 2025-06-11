import json, random, sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


from CausalAbstraction.causal.causal_model import CausalModel
from CausalAbstraction.neural.LM_units import TokenPosition, get_last_token_index

from copy import deepcopy
from tasks.hf_dataloader import load_hf_dataset
import re


def get_causal_model():
    """
    Create and return the causal model for ARC Easy task.
    """
    NUM_CHOICES = 4
    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    variables = ["raw_input", "answer_pointer", "answer", "answerKey", "raw_output"] + ["symbol" + str(x) for x in range(NUM_CHOICES)] 

    values = {}
    values.update({"symbol" + str(x): list(ALPHABET) for x in range(NUM_CHOICES)})
    values.update({"answer_pointer": list(range(NUM_CHOICES)), "answer": list(ALPHABET)})
    values.update({"answerKey": list(range(NUM_CHOICES))})
    # FIXED: Change None to empty list for raw_input and raw_output
    values.update({"raw_input": [""], "raw_output": [""]})

    parents = {"answer":["answer_pointer"] + ["symbol" + str(x) for x in range(NUM_CHOICES)], 
            "answer_pointer": ["answerKey"],
            "answerKey": [],
            "raw_output": ["answer"],
            "raw_input": []}
    parents.update({"symbol" + str(x): [] for x in range(NUM_CHOICES)})

    def get_raw_input():
        return ""

    def get_symbol():
        return random.choice(list(ALPHABET))

    def get_answer_pointer(answerKey):
        return answerKey

    def get_answer(answer_pointer, *symbols):
        return " " + symbols[answer_pointer]

    def get_raw_output(answer):
        return answer

    def get_answerKey():
        return random.choice(list(range(NUM_CHOICES)))

    mechanisms = {
        "raw_input": get_raw_input,
        **{f"symbol{i}": get_symbol for i in range(NUM_CHOICES)},
        "answer_pointer": get_answer_pointer,
        "answer": get_answer,
        "answerKey": get_answerKey,
        "raw_output": get_raw_output
    }

    # Create and initialize the model
    return CausalModel(variables, values, parents, mechanisms, id=f"ARC_easy")


def get_counterfactual_datasets(hf=True, size=None):
    """
    Load and return counterfactual datasets for ARC Easy task.
    """
    # Filter function to only keep examples with exactly 4 choices
    def has_four_choices(example):
        return len(example.get("choices", {}).get("label", [])) == 4
    
    if hf:
        # Load dataset from HuggingFace with customized parsing
        datasets = {}
        for split in ["train", "validation", "test"]:
            temp = load_hf_dataset(
                dataset_path="mib-bench/arc_easy",
                split=split,
                parse_fn=parse_arc_easy_example,
                size=size,
                ignore_names=["symbol"],
                filter_fn=has_four_choices,  # Add filter to only keep 4-choice questions
                shuffle=True  # Shuffle the dataset for better training
            )
            datasets.update(temp)
        
        private = load_hf_dataset(
            dataset_path="mib-bench/arc_easy_private_test",
            split="test",
            parse_fn=parse_arc_easy_example,
            size=size,
            ignore_names=["symbol"],
            filter_fn=has_four_choices,  # Add filter to only keep 4-choice questions
            shuffle=True  # Shuffle the dataset for better training
        )
        datasets.update({k+"private":v for k,v in private.items()})
        
        return datasets
    
    # Non-HF implementation would go here if needed
    # For now, just return empty dict for consistency
    return {}


def parse_arc_easy_example(row):
    """
    Customized parsing function for the ARC Easy dataset.
    Returns a variables dict compatible with the causal model.
    """
    # Get the prompt string
    prompt_str = row.get("prompt", "")
    
    # Create variables dictionary
    variables_dict = {
        "raw_input": prompt_str,
        "answerKey": row["answerKey"]
    }

    # Parse choice labels
    choice_labels = row["choices"]["label"]
    for i in range(len(choice_labels)):
        variables_dict[f"symbol{i}"] = str(choice_labels[i]) 
    
    return variables_dict


def get_token_positions(pipeline, causal_model):
    """
    Get token positions for ARC Easy task interventions.
    """
    def get_correct_symbol_index(input_dict, pipeline, causal_model):
        """
        Find the index of the correct answer symbol token in the prompt.
        
        Args:
            input_dict (dict): The input dictionary to a causal model
            pipeline: The tokenizer pipeline
            causal_model: The causal model
            
        Returns:
            list[int]: List containing the index of the correct answer symbol token
        """
        # Run the model to get the answer position
        output = causal_model.run_forward(input_dict)
        pointer = output["answer_pointer"]
        correct_symbol = output[f"symbol{pointer}"]
        prompt = input_dict["raw_input"]
        
        # Find all single uppercase letters in the prompt
        matches = list(re.finditer(r"\b[A-Z]\b", prompt))
        
        # Find the match corresponding to our correct symbol
        symbol_match = None
        for match in matches:
            if prompt[match.start():match.end()] == correct_symbol:
                symbol_match = match
                break
                
        if not symbol_match:
            raise ValueError(f"Could not find correct symbol {correct_symbol} in prompt: {prompt}")
        
        # Get the substring up to the symbol match end
        substring = prompt[:symbol_match.end()]
        tokenized_substring = list(pipeline.load(substring)["input_ids"][0])
        
        # The symbol token will be at the end of the substring
        return [len(tokenized_substring) - 1]

    # Create TokenPosition objects
    token_positions = [
        TokenPosition(lambda x: get_correct_symbol_index(x, pipeline, causal_model), pipeline, id="correct_symbol"),
        TokenPosition(lambda x: get_last_token_index(x, pipeline), pipeline, id="last_token")
    ]
    return token_positions


def is_unique(lst):
    """Check if all elements in list are unique."""
    return len(lst) == len(set(lst))