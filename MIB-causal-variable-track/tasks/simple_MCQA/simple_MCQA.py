import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


from CausalAbstraction.causal.causal_model import CausalModel
from tasks.hf_dataloader import load_hf_dataset

from copy import deepcopy
from CausalAbstraction.neural.LM_units import TokenPosition, get_last_token_index
import re
import random
import json

def get_causal_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'object_color_pairs.json')
    #Load grandparent directory
    with open(path, 'r') as f:
            data = json.load(f)

    OBJECTS = [item['object'] for item in data]
    COLORS = [item['color'] for item in data]
    COLOR_OBJECTS = [(item["color"], item["object"]) for item in data]

    NUM_CHOICES = 4
    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    variables = ["question", "raw_input"] + ["symbol" + str(x) for x in range(NUM_CHOICES)] + ["choice" + str(x) for x in range(NUM_CHOICES)] + [ "answer_pointer", "answer", "raw_output"]

    values = {"choice" + str(x): COLORS for x in range(NUM_CHOICES)}
    values.update({"symbol" + str(x): ALPHABET for x in range(NUM_CHOICES)})
    values.update({"answer_pointer": range(NUM_CHOICES), "answer": ALPHABET})
    values.update({"question": COLOR_OBJECTS })
    values.update({"raw_input": None, "raw_output": None})

    parents = {"answer":["answer_pointer"] + ["symbol" + str(x) for x in range(NUM_CHOICES)], 
            "answer_pointer": ["question"] + ["choice" + str(x) for x in range(NUM_CHOICES)],
            "raw_output": ["answer"],
            "raw_input": [],
                "question": []}
    parents.update({"choice" + str(x): [] for x in range(NUM_CHOICES)})
    parents.update({"symbol" + str(x): [] for x in range(NUM_CHOICES)})

    def get_question():
        return random.choice(COLOR_OBJECTS)

    def get_symbol():
        return random.choice(list(ALPHABET))

    def get_choice():
        return random.choice(COLORS)

    def get_answer_pointer(question, *choices):
        for i, choice in enumerate(choices):
            if choice == question[0]:
                return i

    def get_answer(answer_pointer, *symbols):
        return " " + symbols[answer_pointer]

    def output_dumper(answer):
        return answer

    mechanisms = {
        "raw_input": lambda: "",
        "question": get_question,
        **{f"symbol{i}": get_symbol for i in range(NUM_CHOICES)},
        
        **{f"choice{i}": get_choice for i in range(NUM_CHOICES)},
        
        "answer_pointer": get_answer_pointer,
        "answer": get_answer,
        "raw_output": output_dumper,
    }


    # Create and initialize the model
    return CausalModel(variables, values, parents, mechanisms, id=f"{NUM_CHOICES}_answer_MCQA")


def get_counterfactual_datasets(hf=True, size=None, load_private_data=False):
    NUM_CHOICES = 4  # Assuming this is fixed at 4 as in the original code
    
    if hf:
        # Load dataset from HuggingFace with customized parsing
        datasets = {}
        for split in ["train", "validation", "test"]:
            temp = load_hf_dataset(
                dataset_path="mib-bench/copycolors_mcqa",
                split=split,
                name=f"{NUM_CHOICES}_answer_choices",
                parse_fn=parse_mcqa_example,
                size=size,
                ignore_names=["noun", "color", "symbol"]
            )
            datasets.update(temp)
        if load_private_data:
            private = load_hf_dataset(
                dataset_path="mib-bench/copycolors_mcqa_private_test",
                split="test",
                name=f"{NUM_CHOICES}_answer_choices",
                parse_fn=parse_mcqa_example,
                size=size,
                ignore_names=["noun", "color", "symbol"]
            )
            datasets.update({k+"private":v for k,v in private.items()})
        
        return datasets
    

def get_token_positions(pipeline, causal_model):
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
        pointer = output["answer_pointer"]
        correct_symbol = output[f"symbol{pointer}"]
        prompt = input["raw_input"]
        
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

    # Create TokenPosition object
    token_positions = [
        TokenPosition(lambda x: get_correct_symbol_index(x, pipeline, causal_model), pipeline, id="correct_symbol"),
        TokenPosition(lambda x: [get_correct_symbol_index(x, pipeline, causal_model)[0]+1], pipeline, id="correct_symbol_period"),
        TokenPosition(lambda x: get_last_token_index(x, pipeline), pipeline, id="last_token")
    ]
    return token_positions


def parse_mcqa_example(row):
    """
    Customized parsing function for the MCQA task.
    Returns a tuple of (prompt_str, variables_dict) like the arithmetic example.
    """
    # Get the prompt/question text
    prompt_str = row.get("prompt", "")

    # Extract object and color information
    q_str = prompt_str
    if " is " in q_str:
        noun, color = q_str.split(" is ", 1)
    elif " are " in q_str:
        noun, color = q_str.split(" are ", 1)
    noun = noun.strip().lower()
    color = color.split(".", 1)[0].strip().lower()

    # Process choices
    choice_labels = row["choices"]["label"]
    choice_texts = row["choices"]["text"]

    # Create the variables dictionary
    variables_dict = {
        "question": (color, noun)
    }

    for i in range(len(choice_labels)):
        variables_dict[f"symbol{i}"] = str(choice_labels[i]) 
        variables_dict[f"choice{i}"] = str(choice_texts[i])
    
    variables_dict["raw_input"] = prompt_str

    # Return tuple of (prompt_str, variables_dict) to match the other file's format
    return variables_dict