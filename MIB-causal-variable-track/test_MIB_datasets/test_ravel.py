import unittest
import sys
import os
import re
from unittest.mock import Mock, MagicMock, patch
from transformers import AutoTokenizer
from CausalAbstraction.neural.pipeline import LMPipeline

# Add the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from tasks.RAVEL.ravel import (
    get_causal_model, 
    get_counterfactual_datasets, 
    parse_ravel_example,
    get_token_positions,
    load_city_entity_data,
    _CITY_ENTITY
)
from datasets import load_dataset


class TestRAVELCausalModel(unittest.TestCase):
    """Test suite for RAVEL causal model validation against HuggingFace dataset."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures with small samples from HF datasets."""
        # Clear and reload city entity data
        _CITY_ENTITY.clear()
        load_city_entity_data()
        
        cls.causal_model = get_causal_model()
        
        # Load small samples from public dataset
        try:
            cls.hf_public_sample = load_dataset(
                "mib-bench/ravel", 
                split="test[:5]",  # Just 5 examples
            )
        except Exception as e:
            print(f"Warning: Could not load public RAVEL dataset: {e}")
            cls.hf_public_sample = []
        
        # Load small samples from private dataset
        try:
            cls.hf_private_sample = load_dataset(
                "mib-bench/ravel_private_test", 
                split="test[:5]",  # Just 5 examples
            )
        except Exception as e:
            print(f"Warning: Could not load private RAVEL dataset: {e}")
            cls.hf_private_sample = []

    def test_city_entity_data_loaded(self):
        """Test that city entity data is loaded properly."""
        self.assertGreater(len(_CITY_ENTITY), 0, "City entity data should be loaded")
        
        # Check structure of city data
        for city, data in _CITY_ENTITY.items():
            self.assertIsInstance(city, str)
            self.assertIsInstance(data, dict)
            self.assertIn("Continent", data)
            self.assertIn("Country", data)
            self.assertIn("Language", data)

    def test_causal_model_variables(self):
        """Test that causal model has all required variables."""
        expected_variables = ["raw_input", "entity", "queried_attribute", "answer", 
                            "raw_output", "Continent", "Country", "Language"]
        
        for var in expected_variables:
            self.assertIn(var, self.causal_model.variables)

    def test_parse_ravel_example_format_public(self):
        """Test that parse_ravel_example returns correct format for public dataset."""
        if not self.hf_public_sample:
            self.skipTest("Public RAVEL dataset not available")
            
        for hf_example in self.hf_public_sample:
            variables_dict = parse_ravel_example(hf_example)
            
            # Check return format
            self.assertIsInstance(variables_dict, dict)
            
            # Check required keys
            self.assertIn('entity', variables_dict)
            self.assertIn('queried_attribute', variables_dict)
            
            # If prompt exists in HF data, check it's preserved
            if 'prompt' in hf_example:
                self.assertIn('raw_input', variables_dict)
                self.assertEqual(variables_dict['raw_input'], hf_example['prompt'])

    def test_parse_ravel_example_format_private(self):
        """Test that parse_ravel_example returns correct format for private dataset."""
        if not self.hf_private_sample:
            self.skipTest("Private RAVEL dataset not available")
            
        for hf_example in self.hf_private_sample:
            variables_dict = parse_ravel_example(hf_example)
            
            # Check return format
            self.assertIsInstance(variables_dict, dict)
            
            # Check required keys
            self.assertIn('entity', variables_dict)
            self.assertIn('queried_attribute', variables_dict)
            
            # If prompt exists in HF data, check it's preserved
            if 'prompt' in hf_example:
                self.assertIn('raw_input', variables_dict)
                self.assertEqual(variables_dict['raw_input'], hf_example['prompt'])

    def test_raw_input_generation(self):
        """Test that causal model generates correct raw_input format."""
        test_cases = [
            ("Paris", "Continent", "Q: What is the continent of Paris? A:"),
            ("Tokyo", "Country", "Q: What is the country of Tokyo? A:"),
            ("London", "Language", "Q: What is the language of London? A:"),
            ("Paris", "wikipedia", "Q: What is Paris? A:"),
        ]
        
        for entity, attribute, expected_prompt in test_cases:
            with self.subTest(entity=entity, attribute=attribute):
                # Create input dict
                input_dict = {
                    "entity": entity,
                    "queried_attribute": attribute
                }
                
                # Run causal model
                output = self.causal_model.run_forward(input_dict)
                
                # Check generated raw_input
                self.assertEqual(output['raw_input'], expected_prompt)

    def test_answer_generation_for_attributes(self):
        """Test that causal model generates correct answers for attribute queries."""
        # Test with known entities
        test_cases = []
        
        # Add test cases for entities we know exist
        if "Paris" in _CITY_ENTITY:
            test_cases.extend([
                ("Paris", "Continent", _CITY_ENTITY["Paris"]["Continent"]),
                ("Paris", "Country", _CITY_ENTITY["Paris"]["Country"]),
                ("Paris", "Language", _CITY_ENTITY["Paris"]["Language"]),
            ])
        
        if "Tokyo" in _CITY_ENTITY:
            test_cases.extend([
                ("Tokyo", "Continent", _CITY_ENTITY["Tokyo"]["Continent"]),
                ("Tokyo", "Country", _CITY_ENTITY["Tokyo"]["Country"]),
                ("Tokyo", "Language", _CITY_ENTITY["Tokyo"]["Language"]),
            ])
        
        for entity, attribute, expected_answer in test_cases:
            with self.subTest(entity=entity, attribute=attribute):
                # Create input dict
                input_dict = {
                    "entity": entity,
                    "queried_attribute": attribute
                }
                
                # Run causal model
                output = self.causal_model.run_forward(input_dict)
                
                # Check answer
                self.assertEqual(output['answer'], expected_answer)

    def test_wikipedia_queries_return_empty(self):
        """Test that wikipedia queries return empty answers."""
        for entity in list(_CITY_ENTITY.keys())[:3]:  # Test first 3 entities
            with self.subTest(entity=entity):
                input_dict = {
                    "entity": entity,
                    "queried_attribute": "wikipedia"
                }
                
                output = self.causal_model.run_forward(input_dict)
                
                # Wikipedia queries should return empty answer
                self.assertEqual(output['answer'], "")
                self.assertEqual(output['raw_output'], "")

    def test_raw_output_format(self):
        """Test that raw_output is correctly formatted."""
        # Test non-wikipedia queries
        if "Paris" in _CITY_ENTITY:
            input_dict = {
                "entity": "Paris",
                "queried_attribute": "Country"
            }
            output = self.causal_model.run_forward(input_dict)
            
            # Should have a space prefix for non-empty answers
            expected_output = f" {_CITY_ENTITY['Paris']['Country']}"
            self.assertEqual(output['raw_output'], expected_output)
        
        # Test wikipedia query
        input_dict = {
            "entity": "Paris",
            "queried_attribute": "wikipedia"
        }
        output = self.causal_model.run_forward(input_dict)
        
        # Should be empty for wikipedia
        self.assertEqual(output['raw_output'], "")

    def test_integration_full_pipeline_public(self):
        """Integration test: full pipeline from HF data through causal model for public dataset."""
        if not self.hf_public_sample:
            self.skipTest("Public RAVEL dataset not available")
            
        for hf_example in self.hf_public_sample:
            # Skip if entity not in our city data
            if 'entity' in hf_example and hf_example['entity'] not in _CITY_ENTITY:
                continue
                
            # Step 1: Parse HF example
            variables_dict = parse_ravel_example(hf_example)
            
            # Step 2: Run causal model
            try:
                causal_output = self.causal_model.run_forward(variables_dict)
                
                # Step 3: Verify output structure
                self.assertIn('raw_input', causal_output)
                self.assertIn('answer', causal_output)
                self.assertIn('raw_output', causal_output)
                
                # Step 4: Verify answer consistency
                entity = variables_dict['entity']
                attribute = variables_dict['queried_attribute']
                
                if attribute in ["Continent", "Country", "Language"]:
                    expected_answer = _CITY_ENTITY[entity][attribute]
                    self.assertEqual(causal_output['answer'], expected_answer)
                elif attribute == "wikipedia":
                    self.assertEqual(causal_output['answer'], "")
                    
            except KeyError:
                # Entity might not be in our test city data
                pass

    def test_integration_full_pipeline_private(self):
        """Integration test: full pipeline from HF data through causal model for private dataset."""
        if not self.hf_private_sample:
            self.skipTest("Private RAVEL dataset not available")
            
        for hf_example in self.hf_private_sample:
            # Skip if entity not in our city data
            if 'entity' in hf_example and hf_example['entity'] not in _CITY_ENTITY:
                continue
                
            # Step 1: Parse HF example
            variables_dict = parse_ravel_example(hf_example)
            
            # Step 2: Run causal model
            try:
                causal_output = self.causal_model.run_forward(variables_dict)
                
                # Step 3: Verify output structure
                self.assertIn('raw_input', causal_output)
                self.assertIn('answer', causal_output)
                self.assertIn('raw_output', causal_output)
                
                # Step 4: Verify answer consistency
                entity = variables_dict['entity']
                attribute = variables_dict['queried_attribute']
                
                if attribute in ["Continent", "Country", "Language"]:
                    expected_answer = _CITY_ENTITY[entity][attribute]
                    self.assertEqual(causal_output['answer'], expected_answer)
                elif attribute == "wikipedia":
                    self.assertEqual(causal_output['answer'], "")
                    
            except KeyError:
                # Entity might not be in our test city data
                pass

    def test_causal_model_values_populated(self):
        """Test that causal model values are properly populated."""
        # Check that values dict has reasonable entries
        self.assertIn("entity", self.causal_model.values)
        self.assertIn("queried_attribute", self.causal_model.values)
        self.assertIn("answer", self.causal_model.values)
        
        # Entity values should match city keys
        self.assertEqual(set(self.causal_model.values["entity"]), set(_CITY_ENTITY.keys()))
        
        # Queried attributes should include the main attributes plus wikipedia
        expected_attributes = ["Continent", "Country", "Language", "wikipedia"]
        self.assertEqual(set(self.causal_model.values["queried_attribute"]), set(expected_attributes))
        
        # Check continent/country/language values are populated
        self.assertGreater(len(self.causal_model.values["Continent"]), 0)
        self.assertGreater(len(self.causal_model.values["Country"]), 0)
        self.assertGreater(len(self.causal_model.values["Language"]), 0)


class TestRAVELTokenPositions(unittest.TestCase):
    """Test suite for RAVEL token position functions."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures with mock pipeline and real tokenizer."""
        # Clear and reload city entity data
        _CITY_ENTITY.clear()
        load_city_entity_data()
        
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
                "mib-bench/ravel", 
                split="test[:10]",
            )
            
            cls.hf_public_sample = list(hf_dataset) if len(hf_dataset) > 0 else []
            
            print(f"Loaded {len(cls.hf_public_sample)} RAVEL HF examples for token position tests")
            
        except Exception as e:
            cls.hf_public_sample = []
            print(f"Warning: Could not load RAVEL HF dataset for token position tests: {e}")

    def test_token_positions_created_correctly(self):
        """Test that token position objects are created with correct IDs."""
        self.assertEqual(len(self.token_positions), 2)
        
        position_ids = [pos.id for pos in self.token_positions]
        self.assertIn("last_token", position_ids)
        self.assertIn("entity_last_token", position_ids)

    def test_entity_last_token_position_synthetic(self):
        """Test entity_last_token position on synthetic examples."""
        entity_last_position = None
        for pos in self.token_positions:
            if pos.id == "entity_last_token":
                entity_last_position = pos
                break
        
        self.assertIsNotNone(entity_last_position)
        
        # Test with synthetic examples using Houston and other cities that should exist
        test_cases = [
            ("Q: What is the continent of Houston? A:", "Houston"),
            ("Q: What is the country of Houston? A:", "Houston"),
            ("Q: What is Houston? A:", "Houston"),
            ("Q: What is the language of Houston? A:", "Houston"),
        ]
        
        for prompt, expected_entity in test_cases:
            with self.subTest(prompt=prompt, entity=expected_entity):
                # Create input dict
                input_dict = {
                    "raw_input": prompt,
                    "entity": expected_entity,
                    "queried_attribute": "Continent"  # Dummy value
                }
                
                try:
                    # Get the token position
                    indices = entity_last_position.index(input_dict)
                    
                    # Should return a list with one index
                    self.assertIsInstance(indices, list)
                    self.assertEqual(len(indices), 1)
                    self.assertIsInstance(indices[0], int)
                    self.assertGreaterEqual(indices[0], 0)
                    
                    # Verify the index is reasonable
                    tokenized = self.mock_pipeline.load(prompt)["input_ids"][0]
                    self.assertLess(indices[0], len(tokenized))
                    
                except Exception as e:
                    # If Houston isn't in the loaded data, skip this test
                    if "Houston" not in _CITY_ENTITY:
                        self.skipTest(f"Houston not found in loaded city data: {e}")
                    else:
                        self.fail(f"Failed to get entity_last_token position: {e}")

    def test_entity_last_token_position_hf_samples(self):
        """Test entity_last_token position on HF dataset samples."""
        if not self.hf_public_sample or len(self.hf_public_sample) == 0:
            self.skipTest("HF dataset not available or empty")
            
        entity_last_position = None
        for pos in self.token_positions:
            if pos.id == "entity_last_token":
                entity_last_position = pos
                break
        
        self.assertIsNotNone(entity_last_position)
        
        for hf_example in self.hf_public_sample[:3]:  # Test first 3 examples
            if not isinstance(hf_example, dict) or "prompt" not in hf_example:
                continue
            
            # Skip if entity not in our city data
            if 'entity' in hf_example and hf_example['entity'] not in _CITY_ENTITY:
                continue
                
            # Parse example to get input dict
            variables_dict = parse_ravel_example(hf_example)
            
            # Add raw_input from prompt if available
            if 'prompt' in hf_example:
                variables_dict['raw_input'] = hf_example['prompt']
            
            with self.subTest(entity=variables_dict.get('entity', 'unknown')):
                try:
                    # Get the token position
                    indices = entity_last_position.index(variables_dict)
                    
                    # Should return a list with one index
                    self.assertIsInstance(indices, list)
                    self.assertEqual(len(indices), 1)
                    self.assertIsInstance(indices[0], int)
                    self.assertGreaterEqual(indices[0], 0)
                    
                    # Verify the index points to a reasonable token
                    if 'raw_input' in variables_dict:
                        tokenized = self.mock_pipeline.load(variables_dict["raw_input"])["input_ids"][0]
                        self.assertLess(indices[0], len(tokenized))
                    
                except Exception as e:
                    # Some examples might not have the entity in our test data
                    pass

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

    def test_highlight_selected_token_functionality(self):
        """Test the highlight_selected_token method works correctly."""
        test_prompt = "Q: What is the continent of Paris? A:"
        
        # Create a variables dict
        variables_dict = {
            "raw_input": test_prompt,
            "entity": "Paris",
            "queried_attribute": "Continent"
        }
        
        for token_pos in self.token_positions:
            with self.subTest(position_id=token_pos.id):
                try:
                    highlighted = token_pos.highlight_selected_token(variables_dict)
                    
                    # Should return a string
                    self.assertIsInstance(highlighted, str)
                    
                    # Should contain the original prompt text
                    self.assertTrue(any(word in highlighted for word in ["Paris", "continent", "?"]))
                    
                    # Should contain highlighting markers
                    self.assertIn("**", highlighted)
                    
                except Exception as e:
                    self.fail(f"Failed to highlight token for {token_pos.id}: {e}")

    def test_entity_position_edge_cases(self):
        """Test entity position with edge cases."""
        entity_last_position = None
        for pos in self.token_positions:
            if pos.id == "entity_last_token":
                entity_last_position = pos
                break
        
        # Test multi-word entities - only use ones we know exist
        multi_word_cases = []
        
        # Check which multi-word entities are actually loaded
        for entity in _CITY_ENTITY.keys():
            if " " in entity:  # Multi-word entity
                if entity == "New York":
                    multi_word_cases.append((f"Q: What is the country of {entity}? A:", entity))
                elif entity == "Los Angeles":
                    multi_word_cases.append((f"Q: What is {entity}? A:", entity))
                elif entity == "São Paulo":
                    multi_word_cases.append((f"Q: What is the language of {entity}? A:", entity))
                else:
                    multi_word_cases.append((f"Q: What is the continent of {entity}? A:", entity))
                
                # Only test first 3 multi-word entities
                if len(multi_word_cases) >= 3:
                    break
        

    def test_entity_not_found_error(self):
        """Test that appropriate error is raised when entity not found."""
        entity_last_position = None
        for pos in self.token_positions:
            if pos.id == "entity_last_token":
                entity_last_position = pos
                break
        
        # Test with prompt that doesn't contain the entity
        input_dict = {
            "raw_input": "Q: What is the continent of London? A:",
            "entity": "Paris",  # Wrong entity
            "queried_attribute": "Continent"
        }
        
        with self.assertRaises(ValueError) as context:
            entity_last_position.index(input_dict)
        
        self.assertIn("not found in prompt", str(context.exception))


class TestRAVELUtilityFunctions(unittest.TestCase):
    """Test suite for RAVEL utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear and reload city entity data
        _CITY_ENTITY.clear()
        load_city_entity_data()

    def test_city_entity_data_structure(self):
        """Test the structure of loaded city entity data."""
        # Check that we have some cities
        self.assertGreater(len(_CITY_ENTITY), 0)
        
        # Check structure of each city
        for city, data in _CITY_ENTITY.items():
            with self.subTest(city=city):
                self.assertIsInstance(data, dict)
                # Required fields for the causal model
                self.assertIn("Continent", data)
                self.assertIn("Country", data)
                self.assertIn("Language", data)
                
                # Check that required values are strings (empty string is ok)
                for key in ["Continent", "Country", "Language"]:
                    self.assertIsInstance(data[key], str, 
                                        f"City '{city}' has non-string value for '{key}': {data[key]}")
                
                # Optional fields that might exist in the JSON (like Latitude, Longitude, Timezone)
                # We don't need to check these as they're not used by the causal model

    @patch('os.path.exists')
    def test_city_entity_fallback(self, mock_exists):
        """Test that city entity data falls back to default when file missing."""
        # Mock file not existing
        mock_exists.return_value = False
        
        # Clear existing data
        _CITY_ENTITY.clear()
        
        # Load data (should use fallback)
        load_city_entity_data()
        
        # Check that we have some default cities
        self.assertGreater(len(_CITY_ENTITY), 0)
        self.assertIn("Paris", _CITY_ENTITY)
        self.assertIn("Tokyo", _CITY_ENTITY)
        self.assertIn("New York", _CITY_ENTITY)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)