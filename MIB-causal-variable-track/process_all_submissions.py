#!/usr/bin/env python3
"""
Process all submission folders by running appropriate evaluation scripts and aggregating results.

This script:
1. Scans a parent directory for submission folders
2. Identifies IOI tasks (folders containing subfolders with "ioi" in the name)
3. Runs ioi_evaluate_submission.py for IOI tasks 
4. Runs evaluate_submission.py for other tasks
5. Aggregates results using aggregate_results.py
6. Saves aggregated results to output folder

Usage:
    python process_all_submissions.py --parent_dir submissions/ --output_dir results/
"""

import os
import sys
import subprocess
import argparse
import json
import shutil
from pathlib import Path


def has_ioi_subfolder(submission_path):
    """
    Check if a submission folder contains any subfolder with 'ioi' in the name.
    
    Args:
        submission_path (str): Path to submission folder
        
    Returns:
        bool: True if IOI subfolder found
    """
    try:
        for item in os.listdir(submission_path):
            item_path = os.path.join(submission_path, item)
            if os.path.isdir(item_path) and 'ioi' in item.lower():
                return True
    except Exception as e:
        print(f"Error checking {submission_path}: {e}")
    return False


def run_command(cmd, description):
    """
    Run a command and capture output.
    
    Args:
        cmd (list): Command and arguments
        description (str): Description of what the command does
        
    Returns:
        bool: True if successful
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        if result.returncode != 0:
            print(f"ERROR: Command failed with return code {result.returncode}")
            return False
            
        return True
    except Exception as e:
        print(f"ERROR running command: {e}")
        return False


def process_submission(submission_path, output_dir, private_data=True):
    """
    Process a single submission folder.
    
    Args:
        submission_path (str): Path to submission folder
        output_dir (str): Directory to save results
        private_data (bool): Whether to evaluate on private data
        
    Returns:
        dict: Results summary
    """
    submission_name = os.path.basename(submission_path)
    print(f"\n{'#'*80}")
    print(f"# Processing submission: {submission_name}")
    print(f"{'#'*80}")
    
    results = {
        "submission": submission_name,
        "has_ioi": False,
        "evaluation_success": False,
        "aggregation_success": False,
        "error": None
    }
    
    # Check if this is an IOI submission
    is_ioi = has_ioi_subfolder(submission_path)
    results["has_ioi"] = is_ioi
    
    # Create output directory for this submission
    submission_output_dir = os.path.join(output_dir, submission_name)
    os.makedirs(submission_output_dir, exist_ok=True)
    
    try:
        if is_ioi:
            # Run IOI evaluation
            print(f"Detected IOI submission - will evaluate using ioi_linear_params.json")
            
            cmd = [
                sys.executable,
                "ioi_evaluate_submission.py",
                "--submission_folder", submission_path
            ]
            
            if private_data:
                cmd.append("--private_data")
            
            success = run_command(cmd, "IOI evaluation")
            results["evaluation_success"] = success
            
        else:
            # Run standard evaluation
            print(f"Detected standard submission - will run evaluate_submission.py")
            
            cmd = [
                sys.executable,
                "evaluate_submission.py",
                "--submission_folder", submission_path
            ]
            
            if private_data:
                cmd.append("--private_data")
            
            success = run_command(cmd, "Standard evaluation")
            results["evaluation_success"] = success
        
        if results["evaluation_success"]:
            # Find all JSON result files in the submission folder
            json_files = []
            for root, dirs, files in os.walk(submission_path):
                for file in files:
                    if file.endswith('.json') and 'results' in file:
                        json_files.append(os.path.join(root, file))
            
            if json_files:
                print(f"\nFound {len(json_files)} result files to aggregate")
                
                # Copy result files to output directory for aggregation
                temp_results_dir = os.path.join(submission_output_dir, "temp_results")
                os.makedirs(temp_results_dir, exist_ok=True)
                
                for json_file in json_files:
                    dest_path = os.path.join(temp_results_dir, os.path.basename(json_file))
                    shutil.copy2(json_file, dest_path)
                
                # Run aggregation
                aggregated_output = os.path.join(submission_output_dir, "aggregated_results.json")
                
                cmd = [
                    sys.executable,
                    "aggregate_results.py",
                    "--folder_path", temp_results_dir,
                    "--output", aggregated_output
                ]
                
                if private_data:
                    cmd.append("--private")
                
                success = run_command(cmd, "Result aggregation")
                results["aggregation_success"] = success
                
                # Clean up temp directory
                shutil.rmtree(temp_results_dir)
                
                # Save summary
                summary_path = os.path.join(submission_output_dir, "processing_summary.json")
                with open(summary_path, 'w') as f:
                    json.dump(results, f, indent=2)
                    
            else:
                print("WARNING: No result files found after evaluation")
                results["error"] = "No result files generated"
                
    except Exception as e:
        print(f"ERROR processing submission: {e}")
        results["error"] = str(e)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Process all submission folders")
    parser.add_argument("--parent_dir", required=True, 
                       help="Parent directory containing submission folders")
    parser.add_argument("--output_dir", required=True,
                       help="Directory to save aggregated results")
    parser.add_argument("--private_data", action="store_true", default=True,
                       help="Evaluate on private test data (default: True)")
    parser.add_argument("--specific_submission", type=str, default=None,
                       help="Process only a specific submission folder")
    
    args = parser.parse_args()
    
    parent_dir = os.path.abspath(args.parent_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    if not os.path.exists(parent_dir):
        print(f"ERROR: Parent directory does not exist: {parent_dir}")
        return 1
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing submissions in: {parent_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Private data evaluation: {args.private_data}")
    
    # Find all submission folders
    submission_folders = []
    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # Skip special directories
            if item in ['__pycache__', '.git', 'results', 'logs']:
                continue
            if args.specific_submission is None or item == args.specific_submission:
                submission_folders.append(item_path)
    
    if not submission_folders:
        print("ERROR: No submission folders found")
        return 1
    
    print(f"\nFound {len(submission_folders)} submission folders to process")
    
    # Process each submission
    all_results = []
    successful = 0
    
    for submission_path in submission_folders:
        results = process_submission(submission_path, output_dir, args.private_data)
        all_results.append(results)
        
        if results["evaluation_success"] and results["aggregation_success"]:
            successful += 1
    
    # Save overall summary
    summary_path = os.path.join(output_dir, "overall_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "total_submissions": len(submission_folders),
            "successful": successful,
            "results": all_results
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE")
    print(f"Successfully processed: {successful}/{len(submission_folders)} submissions")
    print(f"Results saved to: {output_dir}")
    print(f"Overall summary: {summary_path}")
    print(f"{'='*80}")
    
    return 0 if successful == len(submission_folders) else 1


if __name__ == "__main__":
    sys.exit(main())