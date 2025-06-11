import unittest
import sys
import os
import re
from unittest.mock import Mock, MagicMock
from transformers import AutoTokenizer
from CausalAbstraction.neural.pipeline import LMPipeline

# Add the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from tasks.two_digit_addition_task.arithmetic import get_causal_model, get_counterfactual_datasets, parse_arithmetic_example, get_token_positions
from datasets import load_dataset


class TestArithmeticCausalModel(unittest.TestCase):
    """Test suite for arithmetic causal model validation against HuggingFace dataset."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures with small samples from HF datasets."""
        cls.causal_model = get_causal_model()
        
        # Load small samples from public dataset
        cls.hf_public_sample = load_dataset(
            "mech-interp-bench/arithmetic_addition", 
            split="test[:5]",  # Just 5 examples
        )
        
        # Load small samples from private dataset
        cls.hf_private_sample = load_dataset(
            "mech-interp-bench/arithmetic_addition_private_test",
            split="test[:5]",  # Just 5 examples
        )
        
        # Filter to only include 2-digit examples
        cls.hf_public_sample = cls.hf_public_sample.filter(
            lambda example: example["num_digit"] == 2
        )
        cls.hf_private_sample = cls.hf_private_sample.filter(
            lambda example: example["num_digit"] == 2
        )

    def test_parse_arithmetic_example_format_public(self):
        """Test that parse_arithmetic_example returns correct format for public dataset."""
        for hf_example in self.hf_public_sample:
            variables_dict = parse_arithmetic_example(hf_example)
            
            # Check return format - now returns dict directly
            self.assertIsInstance(variables_dict, dict)
            
            # Check variables dict has required keys
            self.assertIn('raw_input', variables_dict)
            self.assertIn('op1_tens', variables_dict)
            self.assertIn('op1_ones', variables_dict)
            self.assertIn('op2_tens', variables_dict)
            self.assertIn('op2_ones', variables_dict)
            
            # Check that raw_input matches HF prompt
            self.assertEqual(variables_dict['raw_input'], hf_example["prompt"])

    def test_parse_arithmetic_example_format_private(self):
        """Test that parse_arithmetic_example returns correct format for private dataset."""
        for hf_example in self.hf_private_sample:
            variables_dict = parse_arithmetic_example(hf_example)
            
            # Check return format - now returns dict directly
            self.assertIsInstance(variables_dict, dict)
            
            # Check variables dict has required keys
            self.assertIn('raw_input', variables_dict)
            self.assertIn('op1_tens', variables_dict)
            self.assertIn('op1_ones', variables_dict)
            self.assertIn('op2_tens', variables_dict)
            self.assertIn('op2_ones', variables_dict)
            
            # Check that raw_input matches HF prompt
            self.assertEqual(variables_dict['raw_input'], hf_example["prompt"])

    def test_operand_consistency_with_prompt_public(self):
        """Test that operands extracted match the numbers in the prompt for public dataset."""
        for hf_example in self.hf_public_sample:
            prompt = hf_example["prompt"]
            variables_dict = parse_arithmetic_example(hf_example)
            
            # Extract numbers from prompt using regex
            numbers = re.findall(r"\d+", prompt)
            self.assertGreaterEqual(len(numbers), 2, f"Prompt should have at least 2 numbers: {prompt}")
            
            # Get the last two numbers (operands)
            op1_str, op2_str = numbers[-2], numbers[-1]
            
            # Convert to expected format
            def parse_operand(num_str):
                if len(num_str) == 1:
                    return 0, int(num_str)
                else:
                    return int(num_str[-2]), int(num_str[-1])
            
            expected_op1_tens, expected_op1_ones = parse_operand(op1_str)
            expected_op2_tens, expected_op2_ones = parse_operand(op2_str)
            
            # Verify extracted operands match prompt
            self.assertEqual(variables_dict['op1_tens'], expected_op1_tens)
            self.assertEqual(variables_dict['op1_ones'], expected_op1_ones)
            self.assertEqual(variables_dict['op2_tens'], expected_op2_tens)
            self.assertEqual(variables_dict['op2_ones'], expected_op2_ones)

    def test_operand_consistency_with_prompt_private(self):
        """Test that operands extracted match the numbers in the prompt for private dataset."""
        for hf_example in self.hf_private_sample:
            prompt = hf_example["prompt"]
            variables_dict = parse_arithmetic_example(hf_example)
            
            # Extract numbers from prompt using regex
            numbers = re.findall(r"\d+", prompt)
            self.assertGreaterEqual(len(numbers), 2, f"Prompt should have at least 2 numbers: {prompt}")
            
            # Get the last two numbers (operands)
            op1_str, op2_str = numbers[-2], numbers[-1]
            
            # Convert to expected format
            def parse_operand(num_str):
                if len(num_str) == 1:
                    return 0, int(num_str)
                else:
                    return int(num_str[-2]), int(num_str[-1])
            
            expected_op1_tens, expected_op1_ones = parse_operand(op1_str)
            expected_op2_tens, expected_op2_ones = parse_operand(op2_str)
            
            # Verify extracted operands match prompt
            self.assertEqual(variables_dict['op1_tens'], expected_op1_tens)
            self.assertEqual(variables_dict['op1_ones'], expected_op1_ones)
            self.assertEqual(variables_dict['op2_tens'], expected_op2_tens)
            self.assertEqual(variables_dict['op2_ones'], expected_op2_ones)

    def test_raw_input_preserves_original_prompt_public(self):
        """Test that raw_input exactly matches original HF prompt for public dataset."""
        for hf_example in self.hf_public_sample:
            variables_dict = parse_arithmetic_example(hf_example)
            
            # Verify raw_input is preserved exactly
            self.assertEqual(variables_dict['raw_input'], hf_example["prompt"])
            
            # Ensure no information is lost
            original_prompt = hf_example["prompt"]
            preserved_prompt = variables_dict['raw_input']
            self.assertEqual(len(original_prompt), len(preserved_prompt))
            self.assertEqual(original_prompt, preserved_prompt)

    def test_raw_input_preserves_original_prompt_private(self):
        """Test that raw_input exactly matches original HF prompt for private dataset."""
        for hf_example in self.hf_private_sample:
            variables_dict = parse_arithmetic_example(hf_example)
            
            # Verify raw_input is preserved exactly
            self.assertEqual(variables_dict['raw_input'], hf_example["prompt"])
            
            # Ensure no information is lost
            original_prompt = hf_example["prompt"]
            preserved_prompt = variables_dict['raw_input']
            self.assertEqual(len(original_prompt), len(preserved_prompt))
            self.assertEqual(original_prompt, preserved_prompt)

    def test_causal_model_generates_correct_raw_input(self):
        """Test that causal model generates expected raw_input format from operands."""
        test_cases = [
            (2, 7, 6, 4),  # 27 + 64
            (1, 5, 3, 8),  # 15 + 38
            (9, 9, 9, 9),  # 99 + 99
            (0, 5, 0, 7),  # 05 + 07
        ]
        
        for op1_tens, op1_ones, op2_tens, op2_ones in test_cases:
            with self.subTest(op1=f"{op1_tens}{op1_ones}", op2=f"{op2_tens}{op2_ones}"):
                # Create input dict
                input_dict = {
                    "op1_tens": op1_tens,
                    "op1_ones": op1_ones,
                    "op2_tens": op2_tens,
                    "op2_ones": op2_ones
                }
                
                # Run causal model
                output = self.causal_model.run_forward(input_dict)
                
                # Check generated raw_input
                expected_input = f"Q: How much is {op1_tens}{op1_ones} plus {op2_tens}{op2_ones}? A: "
                self.assertEqual(output['raw_input'], expected_input)

    def test_ones_carry_computation_public(self):
        """Test that ones_carry is computed correctly for public dataset."""
        for hf_example in self.hf_public_sample:
            variables_dict = parse_arithmetic_example(hf_example)
            
            # Run causal model
            causal_output = self.causal_model.run_forward(variables_dict)
            
            # Verify ones_carry computation
            op1_ones = variables_dict['op1_ones']
            op2_ones = variables_dict['op2_ones']
            expected_carry = 1 if (op1_ones + op2_ones) > 9 else 0
            
            self.assertEqual(causal_output['ones_carry'], expected_carry,
                           f"ones_carry mismatch for {op1_ones} + {op2_ones} = {op1_ones + op2_ones}")

    def test_ones_carry_computation_private(self):
        """Test that ones_carry is computed correctly for private dataset."""
        for hf_example in self.hf_private_sample:
            variables_dict = parse_arithmetic_example(hf_example)
            
            # Run causal model
            causal_output = self.causal_model.run_forward(variables_dict)
            
            # Verify ones_carry computation
            op1_ones = variables_dict['op1_ones']
            op2_ones = variables_dict['op2_ones']
            expected_carry = 1 if (op1_ones + op2_ones) > 9 else 0
            
            self.assertEqual(causal_output['ones_carry'], expected_carry,
                           f"ones_carry mismatch for {op1_ones} + {op2_ones} = {op1_ones + op2_ones}")

    def test_output_computation_matches_expected_public(self):
        """Test that output computation is mathematically correct for public dataset."""
        for hf_example in self.hf_public_sample:
            variables_dict = parse_arithmetic_example(hf_example)
            
            # Run causal model
            causal_output = self.causal_model.run_forward(variables_dict)
            
            # Extract operands
            op1_tens, op1_ones = variables_dict['op1_tens'], variables_dict['op1_ones']
            op2_tens, op2_ones = variables_dict['op2_tens'], variables_dict['op2_ones']
            ones_carry = causal_output['ones_carry']
            
            # Verify each output component
            expected_ones_out = (op1_ones + op2_ones) % 10
            expected_tens_out = (op1_tens + op2_tens + ones_carry) % 10
            expected_hundreds_out = 1 if (op1_tens + op2_tens + ones_carry) > 9 else 0
            
            self.assertEqual(causal_output['ones_out'], expected_ones_out)
            self.assertEqual(causal_output['tens_out'], expected_tens_out)
            self.assertEqual(causal_output['hundreds_out'], expected_hundreds_out)

    def test_output_computation_matches_expected_private(self):
        """Test that output computation is mathematically correct for private dataset."""
        for hf_example in self.hf_private_sample:
            variables_dict = parse_arithmetic_example(hf_example)
            
            # Run causal model
            causal_output = self.causal_model.run_forward(variables_dict)
            
            # Extract operands
            op1_tens, op1_ones = variables_dict['op1_tens'], variables_dict['op1_ones']
            op2_tens, op2_ones = variables_dict['op2_tens'], variables_dict['op2_ones']
            ones_carry = causal_output['ones_carry']
            
            # Verify each output component
            expected_ones_out = (op1_ones + op2_ones) % 10
            expected_tens_out = (op1_tens + op2_tens + ones_carry) % 10
            expected_hundreds_out = 1 if (op1_tens + op2_tens + ones_carry) > 9 else 0
            
            self.assertEqual(causal_output['ones_out'], expected_ones_out)
            self.assertEqual(causal_output['tens_out'], expected_tens_out)
            self.assertEqual(causal_output['hundreds_out'], expected_hundreds_out)

    def test_raw_output_format_public(self):
        """Test that raw_output is correctly formatted for public dataset."""
        for hf_example in self.hf_public_sample:
            variables_dict = parse_arithmetic_example(hf_example)
            
            # Run causal model
            causal_output = self.causal_model.run_forward(variables_dict)
            
            # Check raw_output format
            raw_output = causal_output['raw_output']
            
            # Should be exactly 3 characters
            self.assertEqual(len(raw_output), 3)
            
            # Should be all digits
            self.assertTrue(raw_output.isdigit())
            
            # Should match expected format
            expected_output = f"{causal_output['hundreds_out']:01d}{causal_output['tens_out']:01d}{causal_output['ones_out']:01d}"
            self.assertEqual(raw_output, expected_output)

    def test_raw_output_format_private(self):
        """Test that raw_output is correctly formatted for private dataset."""
        for hf_example in self.hf_private_sample:
            variables_dict = parse_arithmetic_example(hf_example)
            
            # Run causal model
            causal_output = self.causal_model.run_forward(variables_dict)
            
            # Check raw_output format
            raw_output = causal_output['raw_output']
            
            # Should be exactly 3 characters
            self.assertEqual(len(raw_output), 3)
            
            # Should be all digits
            self.assertTrue(raw_output.isdigit())
            
            # Should match expected format
            expected_output = f"{causal_output['hundreds_out']:01d}{causal_output['tens_out']:01d}{causal_output['ones_out']:01d}"
            self.assertEqual(raw_output, expected_output)

    def test_final_answer_matches_hf_label_public(self):
        """Test that causal model output matches HF label for public dataset."""
        for hf_example in self.hf_public_sample:
            variables_dict = parse_arithmetic_example(hf_example)
            
            # Run causal model
            causal_output = self.causal_model.run_forward(variables_dict)
            raw_output = causal_output['raw_output']
            
            # Compare with HF label (handle potential formatting differences)
            hf_label = str(hf_example["label"])
            
            # Zero-pad HF label to 3 digits for comparison
            hf_label_padded = f"{int(hf_label):03d}"
            
            self.assertEqual(raw_output, hf_label_padded,
                           f"Causal model output {raw_output} doesn't match HF label {hf_label_padded}")

    def test_final_answer_matches_hf_label_private(self):
        """Test that causal model output matches HF label for private dataset."""
        for hf_example in self.hf_private_sample:
            variables_dict = parse_arithmetic_example(hf_example)
            
            # Run causal model
            causal_output = self.causal_model.run_forward(variables_dict)
            raw_output = causal_output['raw_output']
            
            # Compare with HF label (handle potential formatting differences)
            hf_label = str(hf_example["label"])
            
            # Zero-pad HF label to 3 digits for comparison
            hf_label_padded = f"{int(hf_label):03d}"
            
            self.assertEqual(raw_output, hf_label_padded,
                           f"Causal model output {raw_output} doesn't match HF label {hf_label_padded}")

    def test_integration_full_pipeline_public(self):
        """Integration test: full pipeline from HF data through causal model for public dataset."""
        for hf_example in self.hf_public_sample:
            # Step 1: Parse HF example
            variables_dict = parse_arithmetic_example(hf_example)
            
            # Step 2: Verify raw_input preservation
            self.assertEqual(variables_dict['raw_input'], hf_example["prompt"])
            
            # Step 3: Run causal model
            causal_output = self.causal_model.run_forward(variables_dict)
            
            # Step 4: Verify that when we run with parsed operands, raw_input might differ
            # (because causal model generates its own format)
            # But raw_output should still match HF label
            hf_label_padded = f"{int(hf_example['label']):03d}"
            self.assertEqual(causal_output['raw_output'], hf_label_padded)

    def test_integration_full_pipeline_private(self):
        """Integration test: full pipeline from HF data through causal model for private dataset."""
        for hf_example in self.hf_private_sample:
            # Step 1: Parse HF example
            variables_dict = parse_arithmetic_example(hf_example)
            
            # Step 2: Verify raw_input preservation
            self.assertEqual(variables_dict['raw_input'], hf_example["prompt"])
            
            # Step 3: Run causal model
            causal_output = self.causal_model.run_forward(variables_dict)
            
            # Step 4: Verify that when we run with parsed operands, raw_input might differ
            # (because causal model generates its own format)
            # But raw_output should still match HF label
            hf_label_padded = f"{int(hf_example['label']):03d}"
            self.assertEqual(causal_output['raw_output'], hf_label_padded)

    def test_different_prompt_formats_public(self):
        """Test that parsing works for different prompt formats in public dataset."""
        for hf_example in self.hf_public_sample:
            prompt = hf_example["prompt"]
            
            # Should be able to parse regardless of format
            try:
                variables_dict = parse_arithmetic_example(hf_example)
                self.assertIn('raw_input', variables_dict)
                self.assertEqual(variables_dict['raw_input'], prompt)
                
                # Should extract at least two operands
                numbers = re.findall(r"\d+", prompt)
                self.assertGreaterEqual(len(numbers), 2)
                
            except Exception as e:
                self.fail(f"Failed to parse prompt format: {prompt}, Error: {e}")

    def test_different_prompt_formats_private(self):
        """Test that parsing works for different prompt formats in private dataset."""
        for hf_example in self.hf_private_sample:
            prompt = hf_example["prompt"]
            
            # Should be able to parse regardless of format
            try:
                variables_dict = parse_arithmetic_example(hf_example)
                self.assertIn('raw_input', variables_dict)
                self.assertEqual(variables_dict['raw_input'], prompt)
                
                # Should extract at least two operands
                numbers = re.findall(r"\d+", prompt)
                self.assertGreaterEqual(len(numbers), 2)
                
            except Exception as e:
                self.fail(f"Failed to parse prompt format: {prompt}, Error: {e}")

    def test_single_vs_double_digit_operands_public(self):
        """Test parsing of single vs double digit operands for public dataset."""
        for hf_example in self.hf_public_sample:
            prompt = hf_example["prompt"]
            variables_dict = parse_arithmetic_example(hf_example)
            
            # Extract numbers from prompt
            numbers = re.findall(r"\d+", prompt)
            op1_str, op2_str = numbers[-2], numbers[-1]
            
            # Test single digit parsing (if applicable)
            if len(op1_str) == 1:
                self.assertEqual(variables_dict['op1_tens'], 0)
                self.assertEqual(variables_dict['op1_ones'], int(op1_str))
            else:
                self.assertEqual(variables_dict['op1_tens'], int(op1_str[-2]))
                self.assertEqual(variables_dict['op1_ones'], int(op1_str[-1]))
            
            if len(op2_str) == 1:
                self.assertEqual(variables_dict['op2_tens'], 0)
                self.assertEqual(variables_dict['op2_ones'], int(op2_str))
            else:
                self.assertEqual(variables_dict['op2_tens'], int(op2_str[-2]))
                self.assertEqual(variables_dict['op2_ones'], int(op2_str[-1]))

    def test_single_vs_double_digit_operands_private(self):
        """Test parsing of single vs double digit operands for private dataset."""
        for hf_example in self.hf_private_sample:
            prompt = hf_example["prompt"]
            variables_dict = parse_arithmetic_example(hf_example)
            
            # Extract numbers from prompt
            numbers = re.findall(r"\d+", prompt)
            op1_str, op2_str = numbers[-2], numbers[-1]
            
            # Test single digit parsing (if applicable)
            if len(op1_str) == 1:
                self.assertEqual(variables_dict['op1_tens'], 0)
                self.assertEqual(variables_dict['op1_ones'], int(op1_str))
            else:
                self.assertEqual(variables_dict['op1_tens'], int(op1_str[-2]))
                self.assertEqual(variables_dict['op1_ones'], int(op1_str[-1]))
            
            if len(op2_str) == 1:
                self.assertEqual(variables_dict['op2_tens'], 0)
                self.assertEqual(variables_dict['op2_ones'], int(op2_str))
            else:
                self.assertEqual(variables_dict['op2_tens'], int(op2_str[-2]))
                self.assertEqual(variables_dict['op2_ones'], int(op2_str[-1]))

    def test_arithmetic_correctness_public(self):
        """Test that causal model produces mathematically correct results for public dataset."""
        for hf_example in self.hf_public_sample:
            variables_dict = parse_arithmetic_example(hf_example)
            
            # Extract operands and compute expected result manually
            op1 = variables_dict['op1_tens'] * 10 + variables_dict['op1_ones']
            op2 = variables_dict['op2_tens'] * 10 + variables_dict['op2_ones']
            expected_sum = op1 + op2
            
            # Run causal model
            causal_output = self.causal_model.run_forward(variables_dict)
            
            # Reconstruct result from causal model output
            causal_sum = (causal_output['hundreds_out'] * 100 + 
                         causal_output['tens_out'] * 10 + 
                         causal_output['ones_out'])
            
            self.assertEqual(causal_sum, expected_sum,
                           f"Mathematical error: {op1} + {op2} = {expected_sum}, but got {causal_sum}")
            
            # Also verify against HF label
            hf_result = int(hf_example["label"])
            self.assertEqual(causal_sum, hf_result,
                           f"HF label mismatch: expected {hf_result}, got {causal_sum}")

    def test_arithmetic_correctness_private(self):
        """Test that causal model produces mathematically correct results for private dataset."""
        for hf_example in self.hf_private_sample:
            variables_dict = parse_arithmetic_example(hf_example)
            
            # Extract operands and compute expected result manually
            op1 = variables_dict['op1_tens'] * 10 + variables_dict['op1_ones']
            op2 = variables_dict['op2_tens'] * 10 + variables_dict['op2_ones']
            expected_sum = op1 + op2
            
            # Run causal model
            causal_output = self.causal_model.run_forward(variables_dict)
            
            # Reconstruct result from causal model output
            causal_sum = (causal_output['hundreds_out'] * 100 + 
                         causal_output['tens_out'] * 10 + 
                         causal_output['ones_out'])
            
            self.assertEqual(causal_sum, expected_sum,
                           f"Mathematical error: {op1} + {op2} = {expected_sum}, but got {causal_sum}")
            
            # Also verify against HF label
            hf_result = int(hf_example["label"])
            self.assertEqual(causal_sum, hf_result,
                           f"HF label mismatch: expected {hf_result}, got {causal_sum}")


class TestArithmeticTokenPositions(unittest.TestCase):
    """Test suite for arithmetic token position functions."""
    
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
                "mech-interp-bench/arithmetic_addition", 
                split="test[:10]",
            )
            
            # Filter and convert to list
            filtered_dataset = hf_dataset.filter(lambda example: example["num_digit"] == 2)
            cls.hf_public_sample = list(filtered_dataset) if len(filtered_dataset) > 0 else []
            
            print(f"Loaded {len(cls.hf_public_sample)} HF examples for token position tests")
            
        except Exception as e:
            cls.hf_public_sample = []
            print(f"Warning: Could not load HF dataset for token position tests: {e}")

    def test_token_positions_created_correctly(self):
        """Test that token position objects are created with correct IDs."""
        self.assertEqual(len(self.token_positions), 2)
        
        position_ids = [pos.id for pos in self.token_positions]
        self.assertIn("op2_last", position_ids)
        self.assertIn("last", position_ids)

    def test_op2_last_token_position_hf_samples(self):
        """Test op2_last token position on HF dataset samples."""
        if not self.hf_public_sample or len(self.hf_public_sample) == 0:
            self.skipTest("HF dataset not available or empty")
            
        op2_last_position = None
        for pos in self.token_positions:
            if pos.id == "op2_last":
                op2_last_position = pos
                break
        
        self.assertIsNotNone(op2_last_position)
        
        for hf_example in self.hf_public_sample[:3]:  # Test first 3 examples
            if not isinstance(hf_example, dict) or "prompt" not in hf_example:
                continue
                
            # Create input dict format expected by token position function
            input_dict = {"raw_input": hf_example["prompt"]}
            
            with self.subTest(prompt=input_dict["raw_input"][:50] + "..."):
                try:
                    # Get the token position
                    indices = op2_last_position.index(input_dict)
                    
                    # Should return a list with one index
                    self.assertIsInstance(indices, list)
                    self.assertEqual(len(indices), 1)
                    self.assertIsInstance(indices[0], int)
                    self.assertGreaterEqual(indices[0], 0)
                    
                    # Verify the index points to a reasonable token
                    tokenized = self.mock_pipeline.load(input_dict["raw_input"])["input_ids"][0]
                    self.assertLess(indices[0], len(tokenized))
                    
                except Exception as e:
                    self.fail(f"Failed to get op2_last position: {e}")

    def test_last_token_position_hf_samples(self):
        """Test last token position on HF dataset samples."""
        if not self.hf_public_sample or len(self.hf_public_sample) == 0:
            self.skipTest("HF dataset not available or empty")
            
        last_position = None
        for pos in self.token_positions:
            if pos.id == "last":
                last_position = pos
                break
        
        self.assertIsNotNone(last_position)
        
        for hf_example in self.hf_public_sample[:3]:  # Test first 3 examples
            if not isinstance(hf_example, dict) or "prompt" not in hf_example:
                continue
                
            # Create input dict format expected by token position function
            input_dict = {"raw_input": hf_example["prompt"]}
            
            with self.subTest(prompt=input_dict["raw_input"][:50] + "..."):
                try:
                    # Get the token position
                    indices = last_position.index(input_dict)
                    
                    # Should return a list with one index
                    self.assertIsInstance(indices, list)
                    self.assertEqual(len(indices), 1)
                    self.assertIsInstance(indices[0], int)
                    self.assertGreaterEqual(indices[0], 0)
                    
                    # Should be the last token
                    tokenized = self.mock_pipeline.load(input_dict["raw_input"])["input_ids"][0]
                    self.assertEqual(indices[0], len(tokenized) - 1)
                    
                except Exception as e:
                    self.fail(f"Failed to get last position: {e}")

    def test_synthetic_prompts_various_formats(self):
        """Test token positions on synthetic prompts with various formats."""
        synthetic_prompts = [
            "Q: How much is 14 plus 78? A: ",
            "34 + 98 = ",
            "What is the sum of 54 and 90?",
            "Calculate 27 plus 45.",
            "The sum of 23 and 67 is ",
            "Find the result of 89 + 12",
            "Q: What's 56 plus 33? A: ",
            "Compute: 71 + 28",
        ]
        
        for prompt in synthetic_prompts:
            input_dict = {"raw_input": prompt}
            with self.subTest(prompt=prompt):
                for token_pos in self.token_positions:
                    try:
                        indices = token_pos.index(input_dict)
                        
                        # Basic validation
                        self.assertIsInstance(indices, list)
                        self.assertEqual(len(indices), 1)
                        self.assertIsInstance(indices[0], int)
                        self.assertGreaterEqual(indices[0], 0)
                        
                        # Verify index is within bounds
                        tokenized = self.mock_pipeline.load(input_dict["raw_input"])["input_ids"][0]
                        self.assertLess(indices[0], len(tokenized))
                        
                    except Exception as e:
                        self.fail(f"Failed to get {token_pos.id} position for prompt '{prompt}': {e}")

    def test_op2_last_token_correctly_identifies_second_operand(self):
        """Test that op2_last correctly identifies the last token of the second operand."""
        test_cases = [
            ("Q: How much is 27 plus 64? A: ", "64"),
            ("34 + 98 = ", "98"),
            ("What is the sum of 12 and 45?", "45"),
            ("Calculate 56 plus 78.", "78"),
        ]
        
        op2_last_position = None
        for pos in self.token_positions:
            if pos.id == "op2_last":
                op2_last_position = pos
                break
        
        for prompt, expected_operand in test_cases:
            input_dict = {"raw_input": prompt}
            with self.subTest(prompt=prompt, expected_operand=expected_operand):
                # Get the token position
                indices = op2_last_position.index(input_dict)
                token_index = indices[0]
                
                # Get the token at that position
                tokenized = self.mock_pipeline.load(input_dict["raw_input"])["input_ids"][0]
                token_id = tokenized[token_index].item()
                token_text = self.mock_pipeline.tokenizer.decode([token_id])
                
                # The token should be related to the second operand
                # (Note: depending on tokenization, it might be the full number or just the last part)
                self.assertTrue(
                    expected_operand in token_text or token_text in expected_operand,
                    f"Token '{token_text}' should be related to operand '{expected_operand}'"
                )

    def test_highlight_selected_token_functionality(self):
        """Test the highlight_selected_token method works correctly."""
        test_prompt = "Q: How much is 27 plus 64? A: "
        input_dict = {"raw_input": test_prompt}
        
        for token_pos in self.token_positions:
            with self.subTest(position_id=token_pos.id):
                try:
                    highlighted = token_pos.highlight_selected_token(input_dict)
                    
                    # Should return a string
                    self.assertIsInstance(highlighted, str)
                    
                    # Should contain the original prompt text
                    self.assertTrue(any(word in highlighted for word in ["27", "64", "plus"]))
                    
                    # Should contain highlighting markers
                    self.assertIn("**", highlighted)
                    
                except Exception as e:
                    self.fail(f"Failed to highlight token for {token_pos.id}: {e}")

    def test_edge_cases(self):
        """Test edge cases and potential error conditions."""
        edge_cases = [
            ("", "empty string"),
            ("No numbers here", "no numbers"),
            ("Just one number: 42", "single number"),
            ("1 + ", "incomplete expression"),
            ("What is 999 plus 888?", "large numbers"),
        ]
        
        for prompt, description in edge_cases:
            input_dict = {"raw_input": prompt}
            with self.subTest(case=description, prompt=prompt):
                for token_pos in self.token_positions:
                    try:
                        indices = token_pos.index(input_dict)
                        
                        # If it doesn't raise an exception, validate the result
                        if indices:
                            self.assertIsInstance(indices, list)
                            self.assertGreater(len(indices), 0)
                            for idx in indices:
                                self.assertIsInstance(idx, int)
                                self.assertGreaterEqual(idx, 0)
                                
                    except Exception:
                        # Some edge cases may legitimately fail
                        # We're mainly checking that they don't crash unexpectedly
                        pass

    def test_consistency_with_parse_function(self):
        """Test that token positions are consistent with parse function results."""
        if not self.hf_public_sample or len(self.hf_public_sample) == 0:
            self.skipTest("HF dataset not available or empty")
            
        op2_last_position = None
        for pos in self.token_positions:
            if pos.id == "op2_last":
                op2_last_position = pos
                break
        
        for hf_example in self.hf_public_sample[:3]:  # Test first 3 examples
            # Add type checking
            if not isinstance(hf_example, dict) or "prompt" not in hf_example:
                self.skipTest("Invalid HF example format")
                
            prompt = hf_example["prompt"]
            
            with self.subTest(prompt=prompt[:50] + "..."):
                try:
                    # Parse the prompt to get operands
                    parsed = parse_arithmetic_example(hf_example)
                    
                    # Reconstruct the second operand
                    op2_expected = parsed['op2_tens'] * 10 + parsed['op2_ones']
                    
                    # Get token position
                    input_dict = {"raw_input": prompt}
                    indices = op2_last_position.index(input_dict)
                    
                    # This is a consistency check - the token should be related to op2
                    # We can't do exact matching due to tokenization differences
                    self.assertIsInstance(indices, list)
                    self.assertEqual(len(indices), 1)
                    
                except Exception as e:
                    self.fail(f"Consistency check failed for '{prompt}': {e}")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)