#!/usr/bin/env python
import os
import re
import json
import pandas as pd
import numpy as np
from openai import OpenAI
import argparse

def simulate_user_query(args):
    """
    Leverage LLM (gpt-4o-mini) to generate a simulated user query for causal discovery and corresponding global_state statistics,
    then generate a fake data set based on the simulated statistics (sample_size, feature_number, and data_type).
    Returns:
        A dictionary with the following keys:
          - "initial_query": the simulated user query string,
          - "statistics": dictionary of simulated statistics,
          - "selected_algorithm": the algorithm chosen in simulation,
          - "fake_data": a pandas DataFrame containing the generated fake dataset.
    """
    # Get organization, project, and API key from environment variables
    organization = args.organization
    project = args.project
    apikey = args.apikey
    
    # Load the high-performance structured meta-prompt from file
    meta_prompt_file = "user_simulation_meta_prompt.txt"
    try:
        with open(meta_prompt_file, "r") as f:
            meta_prompt = f.read()
    except Exception as e:
        meta_prompt = (
            "High-Performance Structured Meta-Prompt for User Simulation:\n"
            "Please generate a JSON object with the required keys as specified."
        )
    
    # Initialize the OpenAI client with the given credentials
    client = OpenAI(organization=organization, project=project, api_key=apikey)
    
    # Call the LLM using the meta-prompt to simulate a user query and statistics
    response = client.chat.completions.create(
         model="gpt-4o",
         messages=[
             {"role": "system", "content": meta_prompt},
             {"role": "user", "content": "Please generate a JSON object with the required keys as specified."}
         ],
         temperature=1.0
    )
    
    # Extract the response content (this is expected to be a JSON string)
    info_extracted = response.choices[0].message.content.strip()
    info_extracted = re.search(r'<json>(.*?)</json>', info_extracted, re.DOTALL).group(1)
    
    try:
        simulation_info = json.loads(info_extracted)
    except Exception as e:
        # Fallback simulated info if LLM parsing fails
        simulation_info = {
            "initial_query": "selected algorithm: FGES; filter: continuous",
            "statistics": {
                "linearity": True,
                "gaussian_error": True,
                "alpha": 0.05,
                "heterogeneous": False,
                "domain_index": None,
                "sample_size": 1000,
                "feature_number": 20,
                "data_type": "Continuous"
            },
        }
    
    # Generate fake data based on the simulated statistics
    stats = simulation_info.get("statistics", {})
    sample_size = stats.get("sample_size", 100)
    feature_number = stats.get("feature_number", 10)
    data_type = stats.get("data_type", "Continuous")
    
    if data_type == "Continuous":
        fake_data = pd.DataFrame(np.random.randn(sample_size, feature_number),
                                 columns=[f"feature_{i}" for i in range(feature_number)])
    elif data_type == "Category":
        fake_data = pd.DataFrame({
            f"feature_{i}": np.random.choice(["A", "B", "C"], size=sample_size)
            for i in range(feature_number)
        })
    elif data_type == "Time-series":
        dates = pd.date_range(start="2020-01-01", periods=sample_size, freq="D")
        fake_data = pd.DataFrame(np.random.randn(sample_size, feature_number),
                                 index=dates,
                                 columns=[f"feature_{i}" for i in range(feature_number)])
    else:
        fake_data = pd.DataFrame(np.random.randn(sample_size, feature_number),
                                 columns=[f"feature_{i}" for i in range(feature_number)])
    
    # Add fake_data to the simulation info dictionary
    simulation_info["fake_data"] = fake_data
    return simulation_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tiny Copilot - Simplified Version")
    parser.add_argument('--data-file', type=str, default="dataset/sachs/sachs.csv", help='Path to dataset file')
    parser.add_argument('--initial_query', type=str, default="selected algorithm: FGES; filter: continuous", help='Initial algorithm query')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode (skip heavy preprocessing)')
    # Modes: normal, skip, simulate
    parser.add_argument('--mode', type=str, choices=["normal", "skip", "simulate"], default="simulate", help='Mode of operation')
    parser.add_argument('--evaluate', action='store_true', default=False, help='Turn on metrics evaluation if ground truth is available')
    # Additional arguments required by the global initialization
    parser.add_argument('--organization', type=str, default="org-Xa9VGT8plP28JBRKtqBRjG5l")
    parser.add_argument('--project', type=str, default="proj_t78xpJomilJJu5qtuOK4vWfR")
    parser.add_argument('--apikey', type=str, default=None)
    parser.add_argument('--simulation_mode', type=str, default="offline")
    parser.add_argument('--data_mode', type=str, default="real")
    parser.add_argument('--demo_mode', type=bool, default=False)

    # Quick test of simulation output
    args = parser.parse_args()
    args.apikey = os.getenv("OPENAI_API_KEY")
    simulated_info = simulate_user_query(args)
    print("Simulated User Query Info:")
    # Use default=str in json.dumps to correctly display non-serializable objects like Timestamp or DataFrame content summary.
    simulated_info['initial_query'] = json.dumps(simulated_info['initial_query'], indent=4, default=str)
    print(simulated_info['initial_query'])
    print(simulated_info['statistics'])
