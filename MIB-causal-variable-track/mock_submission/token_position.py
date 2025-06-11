"""
Token position definitions for MCQA task submission.
This file provides token position functions that identify key tokens in MCQA prompts.
"""

import re
from CausalAbstraction.neural.LM_units import TokenPosition, get_last_token_index


def get_token_positions(pipeline, causal_model):
    """
    Get token positions for the simple MCQA task.
    
    Args:
        pipeline: The language model pipeline with tokenizer
        causal_model: The causal model for the task
        
    Returns:
        list[TokenPosition]: List of TokenPosition objects for intervention experiments
    """
    def get_correct_symbol_index(input, pipeline, causal_model):
        """
        Find the index of the correct answer symbol in the prompt.
        
        Args:
            input (Dict): The input dictionary to a causal model
            pipeline: The tokenizer pipeline
            causal_model: The causal model
            
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

    # Create TokenPosition objects
    token_positions = [
        TokenPosition(lambda x: get_correct_symbol_index(x, pipeline, causal_model), pipeline, id="correct_symbol"),
        TokenPosition(lambda x: [get_correct_symbol_index(x, pipeline, causal_model)[0]+1], pipeline, id="correct_symbol_period"),
        TokenPosition(lambda x: get_last_token_index(x, pipeline), pipeline, id="last_token")
    ]
    return token_positions