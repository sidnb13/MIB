import unittest
import sys
import os
from unittest.mock import Mock, MagicMock
from transformers import AutoTokenizer
from CausalAbstraction.neural.pipeline import LMPipeline

# Add the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from tasks.simple_MCQA.simple_MCQA import get_causal_model, get_counterfactual_datasets, parse_mcqa_example, get_token_positions
from datasets import load_dataset


class TestSimpleMCQACausalModel(unittest.TestCase):
    """Test suite for Simple MCQA causal model validation against HuggingFace dataset."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures with small samples from HF datasets."""
        cls.causal_model = get_causal_model()
        
        # Load small samples from public dataset
        cls.hf_public_sample = load_dataset(
            "mech-interp-bench/copycolors_mcqa", 
            name="4_answer_choices",
            split="test[:5]",  # Just 5 examples
        )
        
        # Load small samples from private dataset
        cls.hf_private_sample = load_dataset(
            "mech-interp-bench/copycolors_mcqa_private_test",
            name="4_answer_choices", 
            split="test[:5]",  # Just 5 examples
        )

    def test_parse_mcqa_example_format_public(self):
        """Test that parse_mcqa_example returns correct format for public dataset."""
        for hf_example in self.hf_public_sample:
            variables_dict = parse_mcqa_example(hf_example)
            
            # Check return format
            self.assertIsInstance(variables_dict, dict)
            
            # Check variables dict has required keys
            self.assertIn('question', variables_dict)
            self.assertIn('raw_input', variables_dict)
            
            # Check question is a tuple
            self.assertIsInstance(variables_dict['question'], tuple)
            self.assertEqual(len(variables_dict['question']), 2)
            
            # Check raw_input matches HF prompt
            self.assertEqual(variables_dict['raw_input'], hf_example["prompt"])
            
            # Check symbols and choices are extracted
            for i in range(len(hf_example["choices"]["label"])):
                self.assertIn(f'symbol{i}', variables_dict)
                self.assertIn(f'choice{i}', variables_dict)
                self.assertEqual(variables_dict[f'symbol{i}'], hf_example["choices"]["label"][i])
                self.assertEqual(variables_dict[f'choice{i}'], hf_example["choices"]["text"][i])

    def test_parse_mcqa_example_format_private(self):
        """Test that parse_mcqa_example returns correct format for private dataset."""
        for hf_example in self.hf_private_sample:
            variables_dict = parse_mcqa_example(hf_example)
            
            # Check return format
            self.assertIsInstance(variables_dict, dict)
            
            # Check variables dict has required keys
            self.assertIn('question', variables_dict)
            self.assertIn('raw_input', variables_dict)
            
            # Check question is a tuple
            self.assertIsInstance(variables_dict['question'], tuple)
            self.assertEqual(len(variables_dict['question']), 2)
            
            # Check raw_input matches HF prompt
            self.assertEqual(variables_dict['raw_input'], hf_example["prompt"])
            
            # Check symbols and choices are extracted
            for i in range(len(hf_example["choices"]["label"])):
                self.assertIn(f'symbol{i}', variables_dict)
                self.assertIn(f'choice{i}', variables_dict)
                self.assertEqual(variables_dict[f'symbol{i}'], hf_example["choices"]["label"][i])
                self.assertEqual(variables_dict[f'choice{i}'], hf_example["choices"]["text"][i])

    def test_answer_pointer_mapping_public(self):
        """Test that correct answer choice maps to answer_pointer for public dataset."""
        for hf_example in self.hf_public_sample:
            # Parse the example to get causal model input
            variables_dict = parse_mcqa_example(hf_example)
            
            # Run causal model
            causal_output = self.causal_model.run_forward(variables_dict)
            
            # Find which choice matches the question color
            question_color = variables_dict['question'][0]
            expected_pointer = None
            
            for i in range(4):
                if f'choice{i}' in variables_dict and variables_dict[f'choice{i}'] == question_color:
                    expected_pointer = i
                    break
            
            # Check that answer_pointer points to the correct choice
            if expected_pointer is not None:
                self.assertEqual(causal_output['answer_pointer'], expected_pointer)

    def test_answer_pointer_mapping_private(self):
        """Test that correct answer choice maps to answer_pointer for private dataset."""
        for hf_example in self.hf_private_sample:
            # Parse the example to get causal model input
            variables_dict = parse_mcqa_example(hf_example)
            
            # Run causal model
            causal_output = self.causal_model.run_forward(variables_dict)
            
            # Find which choice matches the question color
            question_color = variables_dict['question'][0]
            expected_pointer = None
            
            for i in range(4):
                if f'choice{i}' in variables_dict and variables_dict[f'choice{i}'] == question_color:
                    expected_pointer = i
                    break
            
            # Check that answer_pointer points to the correct choice
            if expected_pointer is not None:
                self.assertEqual(causal_output['answer_pointer'], expected_pointer)

    def test_final_answer_matches_expected_public(self):
        """Test that causal model's final answer matches expected answer for public dataset."""
        for hf_example in self.hf_public_sample:
            # Parse the example
            variables_dict = parse_mcqa_example(hf_example)
            
            # Run causal model
            causal_output = self.causal_model.run_forward(variables_dict)
            
            # Find the correct answer: the symbol corresponding to the choice that matches question color
            question_color = variables_dict['question'][0]
            expected_symbol = None
            
            for i in range(4):
                if f'choice{i}' in variables_dict and variables_dict[f'choice{i}'] == question_color:
                    expected_symbol = variables_dict[f'symbol{i}']
                    break
            
            if expected_symbol is not None:
                expected_answer = " " + expected_symbol
                self.assertEqual(causal_output['answer'], expected_answer)

    def test_final_answer_matches_expected_private(self):
        """Test that causal model's final answer matches expected answer for private dataset."""
        for hf_example in self.hf_private_sample:
            # Parse the example
            variables_dict = parse_mcqa_example(hf_example)
            
            # Run causal model
            causal_output = self.causal_model.run_forward(variables_dict)
            
            # Find the correct answer: the symbol corresponding to the choice that matches question color
            question_color = variables_dict['question'][0]
            expected_symbol = None
            
            for i in range(4):
                if f'choice{i}' in variables_dict and variables_dict[f'choice{i}'] == question_color:
                    expected_symbol = variables_dict[f'symbol{i}']
                    break
            
            if expected_symbol is not None:
                expected_answer = " " + expected_symbol
                self.assertEqual(causal_output['answer'], expected_answer)

    def test_raw_output_equals_answer(self):
        """Test that raw_output equals answer in the causal model."""
        for i in range(min(2, len(self.hf_public_sample))):  # Test on first 2 examples
            hf_example = self.hf_public_sample[i]
            variables_dict = parse_mcqa_example(hf_example)
            
            # Run causal model
            causal_output = self.causal_model.run_forward(variables_dict)
            
            # Check that raw_output equals answer
            self.assertEqual(causal_output['raw_output'], causal_output['answer'])

    def test_integration_full_pipeline_public(self):
        """Integration test: full pipeline from HF data through causal model for public dataset."""
        for hf_example in self.hf_public_sample:
            # Step 1: Parse HF example
            variables_dict = parse_mcqa_example(hf_example)
            
            # Step 2: Run causal model
            causal_output = self.causal_model.run_forward(variables_dict)
            
            # Step 3: Verify end-to-end correctness
            # Find which choice matches the question color
            question_color = variables_dict['question'][0]
            expected_pointer = None
            
            for i in range(4):
                if variables_dict[f'choice{i}'] == question_color:
                    expected_pointer = i
                    break
            
            if expected_pointer is not None:
                self.assertEqual(causal_output['answer_pointer'], expected_pointer)
                
                expected_symbol = variables_dict[f'symbol{expected_pointer}']
                expected_answer = " " + expected_symbol
                self.assertEqual(causal_output['answer'], expected_answer)
                
                # Step 4: Verify raw_output
                self.assertEqual(causal_output['raw_output'], expected_answer)

    def test_integration_full_pipeline_private(self):
        """Integration test: full pipeline from HF data through causal model for private dataset."""
        for hf_example in self.hf_private_sample:
            # Step 1: Parse HF example
            variables_dict = parse_mcqa_example(hf_example)
            
            # Step 2: Run causal model
            causal_output = self.causal_model.run_forward(variables_dict)
            
            # Step 3: Verify end-to-end correctness
            # Find which choice matches the question color
            question_color = variables_dict['question'][0]
            expected_pointer = None
            
            for i in range(4):
                if variables_dict[f'choice{i}'] == question_color:
                    expected_pointer = i
                    break
            
            if expected_pointer is not None:
                self.assertEqual(causal_output['answer_pointer'], expected_pointer)
                
                expected_symbol = variables_dict[f'symbol{expected_pointer}']
                expected_answer = " " + expected_symbol
                self.assertEqual(causal_output['answer'], expected_answer)
                
                # Step 4: Verify raw_output
                self.assertEqual(causal_output['raw_output'], expected_answer)

    def test_choice_symbols_extracted_correctly_public(self):
        """Test that choice symbols are correctly extracted from public HF dataset."""
        for hf_example in self.hf_public_sample:
            variables_dict = parse_mcqa_example(hf_example)
            
            # Check that extracted symbols match HF labels
            hf_labels = hf_example["choices"]["label"]
            hf_texts = hf_example["choices"]["text"]
            
            for i in range(len(hf_labels)):
                self.assertEqual(variables_dict[f'symbol{i}'], hf_labels[i])
                self.assertEqual(variables_dict[f'choice{i}'], hf_texts[i])

    def test_choice_symbols_extracted_correctly_private(self):
        """Test that choice symbols are correctly extracted from private HF dataset."""
        for hf_example in self.hf_private_sample:
            variables_dict = parse_mcqa_example(hf_example)
            
            # Check that extracted symbols match HF labels
            hf_labels = hf_example["choices"]["label"]
            hf_texts = hf_example["choices"]["text"]
            
            for i in range(len(hf_labels)):
                self.assertEqual(variables_dict[f'symbol{i}'], hf_labels[i])
                self.assertEqual(variables_dict[f'choice{i}'], hf_texts[i])

    def test_question_parsing_both_formats_public(self):
        """Test that question parsing handles both singular and plural formats for public dataset."""
        for hf_example in self.hf_public_sample:
            variables_dict = parse_mcqa_example(hf_example)
            
            # Check that question tuple has reasonable color and object
            color, obj = variables_dict['question']
            
            # Color and object should be non-empty strings
            self.assertIsInstance(color, str)
            self.assertIsInstance(obj, str)
            self.assertGreater(len(color), 0)
            self.assertGreater(len(obj), 0)
            
            # Color should appear in one of the choices
            choices = [variables_dict[f'choice{i}'] for i in range(4)]
            self.assertIn(color, choices, f"Question color '{color}' should appear in choices {choices}")

    def test_question_parsing_both_formats_private(self):
        """Test that question parsing handles both singular and plural formats for private dataset."""
        for hf_example in self.hf_private_sample:
            variables_dict = parse_mcqa_example(hf_example)
            
            # Check that question tuple has reasonable color and object
            color, obj = variables_dict['question']
            
            # Color and object should be non-empty strings
            self.assertIsInstance(color, str)
            self.assertIsInstance(obj, str)
            self.assertGreater(len(color), 0)
            self.assertGreater(len(obj), 0)
            
            # Color should appear in one of the choices
            choices = [variables_dict[f'choice{i}'] for i in range(4)]
            self.assertIn(color, choices, f"Question color '{color}' should appear in choices {choices}")


class TestSimpleMCQATokenPositions(unittest.TestCase):
    """Test suite for Simple MCQA token position functions."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures with mock pipeline and real tokenizer."""
        cls.causal_model = get_causal_model()
        
        # Create a mock pipeline with real tokenizer but no model
        cls.mock_pipeline = Mock(spec=LMPipeline)
        
        # Load real tokenizer (using a small, fast model)
        real_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        real_tokenizer.pad_token = real_tokenizer.eos_token
        real_tokenizer.pad_token_id = real_tokenizer.convert_tokens_to_ids(real_tokenizer.pad_token)
        
        # Mock the pipeline's tokenizer and load method
        # Mock the pipeline's tokenizer and load method
        cls.mock_pipeline.tokenizer = real_tokenizer

        def mock_load(input_data):
            # Match the real LMPipeline.load() behavior
            if isinstance(input_data, dict):
                if "raw_input" in input_data:
                    raw_input = [input_data["raw_input"]]
                else:
                    raise AssertionError("Input dictionary must contain 'raw_input' key.")
            elif isinstance(input_data, (list, tuple)):
                if all(isinstance(item, dict) and "raw_input" in item for item in input_data):
                    raw_input = [item["raw_input"] for item in input_data]
                elif all(isinstance(item, str) for item in input_data):
                    raw_input = input_data
                else:
                    raise AssertionError("Each input dictionary must contain 'raw_input' key.")
            elif isinstance(input_data, str):
                raw_input = [input_data]
            else:
                raise TypeError("Input must be a dictionary, list/tuple of dictionaries, string, or list of strings.")
            
            return real_tokenizer(
                raw_input,
                padding=True,
                return_tensors="pt",
                add_special_tokens=True
            )

        cls.mock_pipeline.load = mock_load
        
        # Get token positions
        cls.token_positions = get_token_positions(cls.mock_pipeline, cls.causal_model)
        
        # Load HF samples for testing
        try:
            hf_dataset = load_dataset(
                "mech-interp-bench/copycolors_mcqa", 
                name="4_answer_choices",
                split="test[:10]",
            )
            
            cls.hf_public_sample = list(hf_dataset) if len(hf_dataset) > 0 else []
            
            print(f"Loaded {len(cls.hf_public_sample)} MCQA HF examples for token position tests")
            
        except Exception as e:
            cls.hf_public_sample = []
            print(f"Warning: Could not load MCQA HF dataset for token position tests: {e}")

    def test_token_positions_created_correctly(self):
        """Test that token position objects are created with correct IDs."""
        self.assertEqual(len(self.token_positions), 3)
        
        position_ids = [pos.id for pos in self.token_positions]
        self.assertIn("correct_symbol", position_ids)
        self.assertIn("correct_symbol_period", position_ids)
        self.assertIn("last_token", position_ids)

    def test_correct_symbol_token_position_hf_samples(self):
        """Test correct_symbol token position on HF dataset samples."""
        if not self.hf_public_sample or len(self.hf_public_sample) == 0:
            self.skipTest("HF dataset not available or empty")
            
        correct_symbol_position = None
        for pos in self.token_positions:
            if pos.id == "correct_symbol":
                correct_symbol_position = pos
                break
        
        self.assertIsNotNone(correct_symbol_position)
        
        for hf_example in self.hf_public_sample[:3]:  # Test first 3 examples
            if not isinstance(hf_example, dict) or "prompt" not in hf_example:
                continue
                
            # Parse example to get input dict
            variables_dict = parse_mcqa_example(hf_example)
            
            with self.subTest(prompt=variables_dict["raw_input"][:100] + "..."):
                try:
                    # Get the token position
                    indices = correct_symbol_position.index(variables_dict)
                    
                    # Should return a list with one index
                    self.assertIsInstance(indices, list)
                    self.assertEqual(len(indices), 1)
                    self.assertIsInstance(indices[0], int)
                    self.assertGreaterEqual(indices[0], 0)
                    
                    # Verify the index points to a reasonable token
                    tokenized = self.mock_pipeline.load(variables_dict["raw_input"])["input_ids"][0]
                    self.assertLess(indices[0], len(tokenized))
                    
                    # The token should be one of A, B, C, D
                    token_id = tokenized[indices[0]].item()
                    token_text = self.mock_pipeline.tokenizer.decode([token_id]).strip()
                    
                    # Check if it's a valid choice symbol (allowing for tokenizer variations)
                    valid_symbols = {'A', 'B', 'C', 'D'}
                    self.assertTrue(
                        any(symbol in token_text for symbol in valid_symbols),
                        f"Token '{token_text}' should contain a valid choice symbol (A, B, C, D)"
                    )
                    
                except Exception as e:
                    self.fail(f"Failed to get correct_symbol position: {e}")

    def test_last_token_position_hf_samples(self):
        """Test last token position on HF dataset samples."""
        if not self.hf_public_sample or len(self.hf_public_sample) == 0:
            self.skipTest("HF dataset not available or empty")
            
        last_position = None
        for pos in self.token_positions:
            if pos.id == "last_token":
                last_position = pos
                break
        
        self.assertIsNotNone(last_position)
        
        for hf_example in self.hf_public_sample[:3]:  # Test first 3 examples
            if not isinstance(hf_example, dict) or "prompt" not in hf_example:
                continue
                
            # Parse example to get input dict
            variables_dict = parse_mcqa_example(hf_example)
            
            with self.subTest(prompt=variables_dict["raw_input"][:100] + "..."):
                try:
                    # Get the token position
                    indices = last_position.index(variables_dict)
                    
                    # Should return a list with one index
                    self.assertIsInstance(indices, list)
                    self.assertEqual(len(indices), 1)
                    self.assertIsInstance(indices[0], int)
                    self.assertGreaterEqual(indices[0], 0)
                    
                    # Should be the last token
                    tokenized = self.mock_pipeline.load(variables_dict["raw_input"])["input_ids"][0]
                    self.assertEqual(indices[0], len(tokenized) - 1)
                    
                except Exception as e:
                    self.fail(f"Failed to get last position: {e}")

    def test_highlight_selected_token_functionality(self):
        """Test the highlight_selected_token method works correctly."""
        test_prompt = """Question: What color is an apple?
A. red
B. blue
C. green
D. yellow
Answer:"""
        
        # Create a variables dict
        variables_dict = {
            "raw_input": test_prompt,
            "question": ("red", "apple"),
            "symbol0": "A",
            "symbol1": "B",
            "symbol2": "C",
            "symbol3": "D",
            "choice0": "red",
            "choice1": "blue",
            "choice2": "green",
            "choice3": "yellow"
        }
        
        for token_pos in self.token_positions:
            with self.subTest(position_id=token_pos.id):
                try:
                    highlighted = token_pos.highlight_selected_token(variables_dict)
                    
                    # Should return a string
                    self.assertIsInstance(highlighted, str)
                    
                    # Should contain the original prompt text
                    self.assertTrue(any(word in highlighted for word in ["Question", "Answer", "apple"]))
                    
                    # Should contain highlighting markers
                    self.assertIn("**", highlighted)
                    
                except Exception as e:
                    self.fail(f"Failed to highlight token for {token_pos.id}: {e}")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)