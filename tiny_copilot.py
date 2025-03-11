#!/usr/bin/env python
import argparse
import pandas as pd
import os
import json

# Import global state initialization and data loading functions                        
from global_setting.Initialize_state import global_state_initialization, load_data
from preprocess.stat_info_functions import convert_stat_info_to_text
# Import algorithm selection modules
from algorithm.filter import Filter
from algorithm.rerank import Reranker
from algorithm.program import Programming
from algorithm.hyperparameter_selector import HyperparameterSelector

def load_real_world_data(file_path):
    # Baseline: CSV or JSON file loading
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = pd.DataFrame(json.load(f))
    else:
        raise ValueError(f"Unsupported file format for {file_path}")
    
    print("Real-world data loaded successfully.")
    return data

def process_user_query(query, data):
    # Parse the query and filter the data accordingly
    query_dict = {}
    for part in query.split(';'):
        key, value = part.strip().split(':')
        query_dict[key.strip()] = value.strip()

    if 'filter' in query_dict and query_dict['filter'] == 'continuous':
        data = data.select_dtypes(include=['float64', 'int64'])
    
    if 'selected algorithm' in query_dict or 'selected_algorithm' in query_dict:
        selected_algorithm = query_dict.get('selected algorithm', query_dict.get('selected_algorithm'))
        print(f"Algorithm selected: {selected_algorithm}")

    print("User query processed.")
    return data

def simulated_algorithm_run(global_state):
    # Fake simulation (fast-forward) of the programmer module
    print("Simulated algorithm running... (skipping heavy computation)")
    global_state.results.raw_result = "Simulated algorithm selection result"
    return global_state

def main():
    parser = argparse.ArgumentParser(description="Tiny Copilot - Simplified Version")
    parser.add_argument('--data-file', type=str, default="dataset/sachs/sachs.csv", help='Path to dataset file')
    parser.add_argument('--initial_query', type=str, default="Do causal discovery on this biology dataset.", help='Initial algorithm query')
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
  
    args = parser.parse_args()

    # set the api key
    if args.apikey is None:
        args.apikey = os.getenv("OPENAI_API_KEY")

    print(f"Running Tiny Copilot in {args.mode} mode.")

    # Initialize global state    
    if args.mode == "simulate":
        # In simulate mode, use the user_simulation module to generate simulated user queries and statistics
        from algorithm.tests.user_simulation import simulate_user_query
        simulated_info = simulate_user_query(args)
        # Create and populate a new global state with simulated values
        from global_setting.state import GlobalState
        global_state = GlobalState()
        global_state.user_data.initial_query = simulated_info.get("initial_query")
        global_state.statistics.linearity = simulated_info.get("statistics", {}).get("linearity")
        global_state.statistics.gaussian_error = simulated_info.get("statistics", {}).get("gaussian_error")
        global_state.statistics.alpha = simulated_info.get("statistics", {}).get("alpha")
        global_state.statistics.heterogeneous = simulated_info.get("statistics", {}).get("heterogeneous")
        global_state.statistics.domain_index = simulated_info.get("statistics", {}).get("domain_index")
        global_state.statistics.sample_size = simulated_info.get("statistics", {}).get("sample_size")
        global_state.statistics.feature_number = simulated_info.get("statistics", {}).get("feature_number")
        global_state.statistics.data_type = simulated_info.get("statistics", {}).get("data_type")
        global_state.algorithm.selected_algorithm = simulated_info.get("selected_algorithm")
        print("Simulated global state:", simulated_info)
    else:
        # For normal and skip modes, initialize global state from the user's query and file input
        global_state = global_state_initialization(args)
        global_state = load_data(global_state, args)
        
        if args.mode == "normal":
            if args.data_mode == "real":
                global_state.user_data.raw_data = load_real_world_data(args.data_file)
            global_state.user_data.processed_data = process_user_query(args.initial_query, global_state.user_data.raw_data)
        elif args.mode == "skip":
            # Skip preprocessing (debug mode) without modifying the dataset
            if args.data_mode == "real":
                global_state.user_data.raw_data = load_real_world_data(args.data_file)
            global_state.user_data.processed_data = global_state.user_data.raw_data
            # Set statistics for the Sachs dataset, which contains flow cytometry measurements
            # of 11 phosphoproteins and phospholipids in human immune system cells
            global_state.statistics.sample_size = 853  # Number of single cells measured
            if global_state.user_data.processed_data is not None:
                global_state.statistics.feature_number = len(global_state.user_data.processed_data.columns)
            else:
                global_state.statistics.feature_number = 0
            global_state.statistics.missingness = True
            global_state.statistics.data_type = "Continuous"  # Flow cytometry data is continuous
            global_state.statistics.linearity = False  # Protein signaling networks are often nonlinear
            global_state.statistics.gaussian_error = True
            global_state.statistics.stationary = "non time-series"  # Single cell measurements
            global_state.statistics.description = convert_stat_info_to_text(global_state.statistics)
            global_state.user_data.knowledge_docs = "The Sachs dataset contains flow cytometry measurements of 11 phosphoproteins and phospholipids in human immune system cells under different experimental conditions. The data represents a protein signaling network."
        
    # Set selected features if the processed data is available
    if global_state.user_data.processed_data is not None:
        global_state.user_data.visual_selected_features = list(global_state.user_data.processed_data.columns)
        global_state.user_data.selected_features = list(global_state.user_data.processed_data.columns)
    
    print("-"*50, "Global State before algorithm", "-"*50)
    print(global_state)
    print("-"*100)
    
    # --- Start the algorithm selection chain ---
    filter_module = Filter(args)
    global_state = filter_module.forward(global_state)
    
    reranker = Reranker(args)
    global_state = reranker.forward(global_state)

    hp_selector = HyperparameterSelector(args)
    global_state = hp_selector.forward(global_state)

    if args.mode == "simulate":
        # In simulate mode, perform a fast-forward (fake) algorithm run
        global_state = simulated_algorithm_run(global_state)
    else:
        programmer = Programming(args)
        global_state = programmer.forward(global_state)
    
    # Optional metrics evaluation (only if enabled and if ground truth is available)
    if args.evaluate and global_state.user_data.ground_truth is not None:
        from postprocess.judge import Judge
        judge = Judge(global_state, args)
        print("Original Graph: ", global_state.results.converted_graph)
        print("Ground Truth: ", global_state.user_data.ground_truth)
        global_state.results.metrics = judge.evaluation(global_state)
        print("Metrics Evaluation:", global_state.results.metrics)
    
    print("Algorithm selection completed.")
    return global_state

if __name__ == '__main__':
    main()
