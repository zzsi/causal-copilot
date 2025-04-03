import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime
import importlib
import inspect
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from algorithm.evaluation.evaluator import GraphEvaluator
from algorithm.tests.benchmarking_config import get_config

class BenchmarkRunner:
    def __init__(self, algorithm, hyperparams=None, debug_mode: bool = False):
        self.config = get_config()
        self.algorithm = algorithm
        self.hyperparams = hyperparams or {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = self.config['data_dir']
        self.output_dir = os.path.join(self.config['output_dir'], self.timestamp)
        self.evaluator = GraphEvaluator()
        self.debug_mode = debug_mode
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Benchmarking data directory: {self.data_dir}")
        print(f"Benchmarking output directory: {self.output_dir}")
        # Load all algorithm classes from wrappers
        # self.algorithm_candidates = self.config['algorithms']
        self.algorithm_candidates = [self.algorithm]
        self.algorithms = self._load_algorithms()
        
        # Create algorithm identifier with hyperparameters
        self.algo_id = self._create_algo_id()
        
        # Index all available datasets
        self.dataset_index = self._index_datasets()
        
    def _create_algo_id(self) -> str:
        """Create a unique identifier for algorithm with hyperparameters"""
        if not self.hyperparams:
            return self.algorithm
            
        # Create a string representation of the hyperparameters
        param_str = "_".join([f"{k}={v}" for k, v in sorted(self.hyperparams.items())])
        return f"{self.algorithm}_{param_str}"

    def _load_algorithms(self) -> Dict:
        """Dynamically load all algorithm classes from wrappers directory"""
        algorithms = {}
        wrappers_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'wrappers')
        
        # Import all algorithm classes from __init__.py
        import algorithm.wrappers as wrappers
        
        # Get all available algorithm classes
        available_algorithms = {}
        for name in self.algorithm_candidates: # wrappers.__all__:
            available_algorithms[name] = getattr(wrappers, name)
            
        # Only keep algorithms specified in config
        for algo_name in self.algorithm_candidates:
            if algo_name in available_algorithms:
                algorithms[algo_name] = available_algorithms[algo_name]
                # print(f"Added algorithm: {algo_name}")
        
        print(f"Loaded {len(algorithms)} algorithms, {' | '.join(list(algorithms.keys()))}")
                        
        return algorithms

    
    def _index_datasets(self) -> List[Tuple[str, str, Dict]]:
        """Index all available datasets, including absolute paths and configs"""
        dataset_index = []
        
        # Check if the data directory exists
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} does not exist")
            return dataset_index
            
        # Recursively walk through all directories
        for root, dirs, files in os.walk(self.data_dir):
            # Check if this directory contains dataset files (config.json)
            config_files = [f for f in files if f.endswith('config.json') and f != 'simulation_config.json' and f != 'dataset_configs.json']
            if not config_files:
                continue
                
            # Process each config file in the directory
            for config_file in config_files:
                config_path = os.path.join(root, config_file)
                dataset_dir_abs = os.path.dirname(config_path)
                
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Extract seed number using regex
                    import re
                    seed_match = re.search(r'seed_(\d+)', config_path)
                    seed_num = int(seed_match.group(1)) if seed_match else 0
                    config['seed'] = seed_num
                    
                    # Create dataset name with key configuration parameters
                    dataset_name = f"nodes{config.get('n_nodes', 'unknown')}_samples{config.get('n_samples', 'unknown')}_domains{config.get('n_domains', 'unknown')}_noise{config.get('noise_type', 'unknown')}_func{config.get('function_type', 'unknown')}"
                    
                    if 'edge_probability' in config and config['edge_probability'] is not None:
                        dataset_name += f"_edgeprob{config['edge_probability']}"
                    if 'experiment_type' in config:
                        dataset_name = f"{config['experiment_type']}_{dataset_name}"
                    if 'discrete_ratio' in config and config['discrete_ratio'] is not None:
                        dataset_name += f"_discrete{config['discrete_ratio']}"
                    if 'measurement_error' in config and config['measurement_error'] is not None:
                        measurement_error = config['measurement_error_value']
                        dataset_name += f"_measurement{measurement_error}"
                    if 'missing_rate' in config and config['missing_rate'] is not None:
                        missing_rate = config['missing_rate_value']
                        dataset_name += f"_missing{missing_rate}"
                    
                    dataset_name += f"_seed{seed_num}"
                    
                    # Add scenario/category information from directory structure
                    rel_path = os.path.relpath(dataset_dir_abs, self.data_dir)
                    path_parts = rel_path.split(os.sep)
                    if len(path_parts) > 1 and path_parts[0] != '.':
                        scenario = path_parts[0]
                        if len(path_parts) > 2:
                            scenario += f"/{path_parts[1]}"
                        dataset_name = f"{scenario}_{dataset_name}"
                    
                    dataset_index.append((dataset_name, dataset_dir_abs, config))
                    
                except Exception as e:
                    print(f"Error processing config file {config_path}: {e}")
        
        print(f"Found {len(dataset_index)} datasets")
        
        # Sort by n_nodes first, then by n_samples
        dataset_index.sort(key=lambda x: (
            x[2].get('n_nodes', 0), 
            x[2].get('n_samples', 0)
        ))
            
        return dataset_index

    def _load_dataset(self, dataset_dir: str) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
        """Load data, true graph, and config from a dataset directory"""
        data_file = [file for file in os.listdir(dataset_dir) if 'data.csv' in file][0]
        data = pd.read_csv(os.path.join(dataset_dir, data_file))
        true_graph = [file for file in os.listdir(dataset_dir) if 'graph.npy' in file][0]
        true_graph = np.load(os.path.join(dataset_dir, true_graph))
        config_file = [file for file in os.listdir(dataset_dir) if 'config.json' in file][0]
        config = json.load(open(os.path.join(dataset_dir, config_file)))

        return data, true_graph, config

    def _run_single_experiment(self, algo_instance: Any, data: pd.DataFrame, 
                             true_graph: np.ndarray, config: Dict) -> Dict:
        """Run a single experiment with given algorithm and configuration"""
        start_time = time.time()
        
        # Set a timeout limit (10 minutes)
        timeout_seconds = 600
        
        try:
            # fill the missing value if it is existed
            data = data.fillna(0)

            # Skip if the algorithm is not designed for heterogeneous data
            if 'domain_index' in data.columns and "CDNOD" not in algo_instance.name:
                data = data.drop(columns=['domain_index'])
            if 'domain_index' not in data.columns and "CDNOD" in algo_instance.name:
                data["domain_index"] = np.ones(data.shape[0])

            # Use signal-based timeout for Unix systems
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Execution timed out after {timeout_seconds} seconds")
            
            # Set the timeout handler
            original_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            # Run the algorithm
            adj_matrix, info, _ = algo_instance.fit(data)
            
            # Disable the alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)
            
            runtime = time.time() - start_time
            metrics = self.evaluator.compute_metrics(true_graph, adj_matrix)
            
            result = {
                "success": True,
                "runtime": runtime,
                "metrics": metrics,
                "algorithm_params": algo_instance.get_params(),
                "data_config": config
            }
            
        except TimeoutError as e:
            result = {
                "success": False,
                "error": str(e),
                "data_config": config
            }
        except Exception as e:
            result = {
                "success": False,
                "error": str(e),
                "data_config": config
            }
            # Ensure we disable the alarm in case of other exceptions
            try:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, original_handler)
            except:
                pass
            
        return result
    
    
    def _save_result(self, algo_id: str, result: Dict) -> None:
        """Save the result of a single experiment to a JSON file"""
        output_file = os.path.join(self.output_dir, f"{algo_id}_results.json")
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
        else:
            existing_results = []
        
        existing_results.append(result)
        
        with open(output_file, 'w') as f:
            json.dump(existing_results, f, indent=2, default=str)

    def run_all_experiments(self) -> None:
        """Run all benchmark experiments"""
        all_results = {}
        
        # Track current node size and configuration to implement early stopping
        current_node_size = -1
        timeout_config = {}  # Store configurations where timeout occurred
        
        # Run experiments for all datasets
        for dataset_name, dataset_dir, config in tqdm(self.dataset_index, desc=f"Running {len(self.dataset_index)} experiments"):
            # Extract key configuration parameters
            node_size = config['n_nodes']
            function_type = config.get('function_type', 'unknown')
            edge_probability = config.get('edge_probability', 0)
            
            # Create a config signature to track timeouts more precisely
            config_signature = f"nodes{node_size}_func{function_type}_edgeprob{edge_probability}"
            
            # If we're moving to a new node size, update tracking
            if node_size > current_node_size:
                current_node_size = node_size
            
            # Skip if we've already had a timeout for this specific configuration
            if config_signature in timeout_config:
                print(f"Skipping {dataset_name} as timeout occurred for similar configuration: {config_signature}")
                
                # Create a skipped result entry
                skip_result = {
                    "success": False,
                    "error": f"Skipped due to previous timeout with similar configuration: {config_signature}",
                    "data_config": config,
                    "algorithm": next(iter(self.algorithms.keys())),  # Get first algorithm name
                    "hyperparams": self.hyperparams,
                    "algorithm_id": self.algo_id,
                    "dataset": dataset_name
                }
                if self.algo_id not in all_results:
                    all_results[self.algo_id] = []
                all_results[self.algo_id].append(skip_result)
                self._save_result(self.algo_id, skip_result)
                continue
            
            data, true_graph, _ = self._load_dataset(dataset_dir)
            
            for algo_name, algo_class in self.algorithms.items():
                # Initialize algorithm with hyperparameters
                algo_instance = algo_class(self.hyperparams)
                print(f"Running {self.algo_id} on {dataset_name}")
                result = self._run_single_experiment(algo_instance, data, true_graph, config)
                result['algorithm'] = algo_name
                result['hyperparams'] = self.hyperparams
                result['algorithm_id'] = self.algo_id
                result['dataset'] = dataset_name
                if self.algo_id not in all_results:
                    all_results[self.algo_id] = []
                all_results[self.algo_id].append(result)
                self._save_result(self.algo_id, result)
                
                # Check if timeout occurred and mark this configuration
                if not result['success'] and 'timeout' in str(result.get('error', '')).lower():
                    timeout_config[config_signature] = True
                    print(f"Timeout occurred for configuration {config_signature}. Will skip remaining datasets with similar configuration.")
                
            if self.debug_mode:
                break  # Only test the first dataset in debug mode
                
        print(f"Benchmark results saved to: {self.output_dir}")

def test_timeout_mechanism():
    """Test case to verify the timeout kill mechanism works properly"""
    import signal
    
    # Define a function that will run for longer than the timeout
    def long_running_function():
        print("Starting long-running function...")
        time.sleep(10)  # This should be killed by the timeout
        print("This should not be printed!")
        return "Completed"
    
    # Set timeout to 2 seconds
    timeout_seconds = 2
    
    # Set up the timeout handler
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Execution timed out after {timeout_seconds} seconds")
    
    # Set the timeout handler
    original_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, timeout_handler)
    
    try:
        print(f"Setting timeout to {timeout_seconds} seconds...")
        signal.alarm(timeout_seconds)
        
        # Run the function that should be killed
        result = long_running_function()
        
        # If we get here, the timeout didn't work
        print("ERROR: Timeout mechanism failed!")
        return False
        
    except TimeoutError as e:
        print(f"SUCCESS: Timeout mechanism worked correctly: {e}")
        return True
    finally:
        # Disable the alarm and restore original handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--params", type=str, help="JSON string of hyperparameters")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test-timeout", action="store_true", help="Test the timeout mechanism")
    args = parser.parse_args()
    
    if args.test_timeout:
        test_timeout_mechanism()
        return
    
    # Parse hyperparameters if provided
    hyperparams = {}
    if args.params:
        try:
            hyperparams = json.loads(args.params)
        except json.JSONDecodeError:
            print("Error: Could not parse hyperparameters JSON string")
            sys.exit(1)
    
    runner = BenchmarkRunner(algorithm=args.algorithm, hyperparams=hyperparams, debug_mode=args.debug)
    runner.run_all_experiments()

if __name__ == "__main__":
    main()
