#!/bin/bash

# Script to run all baseline scripts with --quick_test flag
# This runs minimal tests with reduced dataset sizes and layers

echo "==============================================="
echo "Running all baselines with --quick_test flag"
echo "==============================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to run a command and check its status
run_test() {
    local name=$1
    shift  # Remove the first argument (name) from $@
    
    echo -e "\n${GREEN}Running $name...${NC}"
    echo "Command: $@"
    
    if "$@"; then
        echo -e "${GREEN}✓ $name completed successfully${NC}"
    else
        echo -e "${RED}✗ $name failed${NC}"
        exit 1
    fi
}

# Change to baselines directory
cd "$(dirname "$0")"

# Run simple MCQA baseline
run_test "Simple MCQA Baseline" python simple_MCQA_baselines.py --quick_test

# Run ARC baseline
run_test "ARC Baseline" python ARC_baselines.py --quick_test

# Run arithmetic baseline
run_test "Arithmetic Baseline" python arithmetic_baselines.py --quick_test

# Run RAVEL baseline
run_test "RAVEL Baseline" python ravel_baselines.py --quick_test

# Run IOI linear parameter learning (needed for IOI baseline)
echo -e "\n${GREEN}Preparing for IOI baseline...${NC}"
run_test "IOI Linear Parameter Learning (GPT2)" python ioi_baselines/ioi_learn_linear_params.py --model gpt2 --quick_test --output_file ioi_linear_params_quick.json
# Run IOI baseline with the computed parameters
run_test "IOI Baseline" python ioi_baselines/ioi_baselines.py --model gpt2 --linear_params ioi_linear_params_quick.json --quick_test --run_baselines --run_brute_force

run_test "IOI Linear Parameter Learning (llama)" python ioi_baselines/ioi_learn_linear_params.py --model llama --quick_test --output_file ioi_linear_params_quick.json
# Run IOI baseline with the computed parameters
run_test "IOI Baseline" python ioi_baselines/ioi_baselines.py --model llama --linear_params ioi_linear_params_quick.json --quick_test --run_baselines --run_brute_force

run_test "IOI Linear Parameter Learning (gemma)" python ioi_baselines/ioi_learn_linear_params.py --model gemma --quick_test --output_file ioi_linear_params_quick.json
Run IOI baseline with the computed parameters
run_test "IOI Baseline" python ioi_baselines/ioi_baselines.py --model gemma --linear_params ioi_linear_params_quick.json --quick_test --run_baselines --run_brute_force --heads_list "(3, 1)" "(4, 2)"

echo -e "\n${GREEN}===============================================${NC}"
echo -e "${GREEN}All quick tests completed successfully!${NC}"
echo -e "${GREEN}===============================================${NC}"