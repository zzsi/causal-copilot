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
    def __init__(self, algorithm, debug_mode: bool = False):
        self.config = get_config()
        self.algorithm = algorithm
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
        
        # Index all available datasets
        self.dataset_index = self._index_datasets()

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
    
        # heuristic: if there is a .json file in the dataset directory, it is a dataset instead of a subdirectory
        if any('.json' in file for file in os.listdir(self.data_dir)):
            dataset_dirs = [self.data_dir]
        else:
            dataset_dirs = os.listdir(self.data_dir)
        print(f"Found {len(dataset_dirs)} datasets")
            
        # First collect all datasets with their configs
        for dataset_dir in dataset_dirs:
            if len(dataset_dirs) == 1:
                dataset_dir_abs = self.data_dir
            else:
                dataset_dir_abs = os.path.join(self.data_dir, dataset_dir)
            config_file = [file for file in os.listdir(dataset_dir_abs) if 'config.json' in file][0]
            config_file = os.path.join(dataset_dir_abs, config_file)
            if not os.path.exists(config_file):
                continue
                
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Extract seed number using regex
            import re
            seed_match = re.search(r'seed_(\d+)', dataset_dir)
            seed_num = int(seed_match.group(1)) if seed_match else 0
            config['seed'] = seed_num
            
            # Create dataset name with key configuration parameters
            dataset_name = f"nodes{config['n_nodes']}_samples{config['n_samples']}_domains{config['n_domains']}_noise{config['noise_type']}_func{config['function_type']}"
            if 'edge_probability' in config and config['edge_probability'] is not None:
                dataset_name += f"_edgeprob{config['edge_probability']}"
            if 'experiment_type' in config:
                dataset_name = f"{config['experiment_type']}_{dataset_name}"
            
            if 'discrete_ratio' in config and config['discrete_ratio'] is not None:
                dataset_name += f"_discrete{config['discrete_ratio']}"
            if 'measurement_error' in config and config['measurement_error'] is not None:
                dataset_name += f"_measurement{config['measurement_error']}"
            if 'missing_rate' in config and config['missing_rate'] is not None:
                dataset_name += f"_missing{config['missing_rate']}"
            
            dataset_name += f"_seed{seed_num}"
                
            dataset_index.append((dataset_name, dataset_dir_abs, config))
        
        # Sort by n_nodes first, then by n_samples
        dataset_index.sort(key=lambda x: (x[2]['n_nodes'], x[2]['n_samples']))
            
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
        
        # Create an event to signal timeout
        import threading
        import multiprocessing as mp
        timeout_event = threading.Event()
        
        # Function to monitor execution time
        def monitor_timeout():
            if time.time() - start_time > 1800:  # 30 minutes = 1800 seconds
                timeout_event.set()
                
        # Start the monitoring thread
        monitor_thread = threading.Thread(target=monitor_timeout)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            # fill the missing value if it is existed
            data = data.fillna(0)

            # Skip if the algorithm is not designed for heterogeneous data
            if 'domain_index' in data.columns and "CDNOD" not in algo_instance.name:
                data = data.drop(columns=['domain_index'])
            if 'domain_index' not in data.columns and "CDNOD" in algo_instance.name:
                # independent domain index randomly generated
                data["domain_index"] = np.random.randn(data.shape[0])

            # Check for timeout periodically during execution
            adj_matrix, info, _ = algo_instance.fit(data)
            
            # Check if timeout occurred
            if timeout_event.is_set():
                return {
                    "success": False,
                    "error": "Execution timed out after 30 minutes",
                    "data_config": config
                }
                
            runtime = time.time() - start_time
            metrics = self.evaluator.compute_metrics(true_graph, adj_matrix)
            
            result = {
                "success": True,
                "runtime": runtime,
                "metrics": metrics,
                "algorithm_params": algo_instance.get_params(),
                "data_config": config
            }
            
        except Exception as e:
            result = {
                "success": False,
                "error": str(e),
                "data_config": config
            }
            
        # Check one more time for timeout
        if timeout_event.is_set():
            result = {
                "success": False,
                "error": "Execution timed out after 30 minutes",
                "data_config": config
            }
            
        return result
    def _save_result(self, algo_name: str, result: Dict) -> None:
        """Save the result of a single experiment to a JSON file"""
        output_file = os.path.join(self.output_dir, f"{algo_name}_results.json")
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
        
        # Run experiments for all datasets
        for dataset_name, dataset_dir, config in tqdm(self.dataset_index, desc=f"Running {len(self.dataset_index)} experiments"):
            data, true_graph, _ = self._load_dataset(dataset_dir)
            
            for algo_name, algo_class in self.algorithms.items():
                algo_instance = algo_class()
                print(f"Running {algo_name} on {dataset_name}")
                result = self._run_single_experiment(algo_instance, data, true_graph, config)
                result['algorithm'] = algo_name
                result['dataset'] = dataset_name
                if algo_name not in all_results:
                    all_results[algo_name] = []
                all_results[algo_name].append(result)
                self._save_result(algo_name, result)
                
            if self.debug_mode:
                break  # Only test the first dataset in debug mode
                
        print(f"Benchmark results saved to: {self.output_dir}")

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    runner = BenchmarkRunner(algorithm=args.algorithm, debug_mode=args.debug)
    runner.run_all_experiments()

if __name__ == "__main__":
    main()
