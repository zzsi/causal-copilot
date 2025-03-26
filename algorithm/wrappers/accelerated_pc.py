import numpy as np
import pandas as pd
from typing import Dict, Tuple
# import psutil
# import GPUtil
import time

# use the local causal-learn package
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.append(root_dir)
sys.path.append(algorithm_dir)

from externals.acceleration.pc.pc import accelerated_pc

from algorithm.wrappers.pc import PC
from algorithm.evaluation.evaluator import GraphEvaluator
from causallearn.graph.GraphClass import CausalGraph, GeneralGraph

class AcceleratedPC(PC):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 0.05,
            'indep_test': 'fisherz',
            'depth': 3, # -1,
            'gamma': 1.0
        }
        self._params.update(params)

    @property
    def name(self):
        return "AcceleratedPC"

    def get_primary_params(self):
        self._primary_param_keys = ['alpha', 'indep_test', 'depth']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['gamma']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, CausalGraph]:
        data_values = data.values

        # Combine primary and secondary parameters
        all_params = {**self.get_primary_params(), **self.get_secondary_params()}

        # Run PC algorithm
        adj_matrix, info, cg = accelerated_pc(data_values, **all_params)

        # Convert the graph to adjacency matrix
        adj_matrix = self.convert_to_adjacency_matrix(cg)

        # Prepare additional information
        info = {
            'sepset': cg.sepset,
            'definite_UC': cg.definite_UC,
            'definite_non_UC': cg.definite_non_UC,
            'PC_elapsed': cg.PC_elapsed,
        }

        return adj_matrix, info, cg
    
    def convert_to_adjacency_matrix(self, cg: CausalGraph) -> np.ndarray:
        adj_matrix = cg.G
        inferred_flat = np.zeros_like(adj_matrix)
        indices = np.where(adj_matrix == 1)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == -1:
                # directed edge: j -> i
                inferred_flat[i, j] = 1
            elif adj_matrix[j, i] == 1:
                # bidirected edge: j <-> i
                if inferred_flat[j, i] == 0:
                    # keep asymmetric that only one entry is recorded
                    inferred_flat[i, j] = 3

        indices = np.where(adj_matrix == -1)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == -1:
                # undirected edge: j -- i
                if inferred_flat[j, i] == 0:
                    inferred_flat[i, j] = 2
        return inferred_flat

    def test_algorithm(self):
        # Generate sample data with linear relationships
        np.random.seed(42)
        n_samples = 2000
        X1 = np.random.normal(0, 1, n_samples)
        X2 = 0.5 * X1 + np.random.normal(0, 0.5, n_samples)
        X3 = 0.3 * X1 + 0.7 * X2 + np.random.normal(0, 0.3, n_samples)
        X4 = 0.6 * X2 + np.random.normal(0, 0.4, n_samples)
        X5 = 0.4 * X3 + 0.5 * X4 + np.random.normal(0, 0.2, n_samples)
        
        df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5})

        print("Testing PC algorithm with pandas DataFrame:")
        params = {
            'alpha': 0.05,
            'depth': 2,
            'indep_test': 'fisherz',
            'verbose': False,
            'show_progress': False
        }
        adj_matrix, info, _ = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)
        print("\nAdditional Info:")
        print(f"PC elapsed time: {info['PC_elapsed']:.4f} seconds")
        print(f"Number of definite unshielded colliders: {len(info['definite_UC'])}")
        print(f"Number of definite non-unshielded colliders: {len(info['definite_non_UC'])}")

        # Ground truth graph
        gt_graph = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0]
        ])

        # Use GraphEvaluator to compute metrics
        evaluator = GraphEvaluator()
        metrics = evaluator.compute_metrics(gt_graph, adj_matrix)

        print("\nMetrics:")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"SHD: {metrics['shd']:.4f}")

    # def stress_test(self, max_vars=100000, max_samples=100000, step_vars=1000, step_samples=10000):
    #     """
    #     Run stress test with increasing variables and samples to monitor GPU memory usage
    #     """
    #     print("\nRunning Stress Test...")
    #     print("=" * 80)
    #     print(f"{'Vars':>5} {'Samples':>8} {'GPU Mem (MB)':>12} {'Time (s)':>10} {'Status':>10}")
    #     print("-" * 80)

    #     # Create DataFrame to store results
    #     import queue
    #     import threading
    #     results = []
    #     # Create a thread-safe queue to store memory measurements
    #     memory_queue = queue.Queue()
        
    #     # def monitor_memory():
    #     #     """Monitor GPU memory usage in a separate thread"""
    #     #     while True:
    #     #         try:
    #     #             gpu = GPUtil.getGPUs()[0]
    #     #             memory_queue.put(gpu.memoryUsed)
    #     #             time.sleep(0.01)  # Check every 10ms
    #     #         except:
    #     #             break

    #     # Record initial memory
    #     # gpu = GPUtil.getGPUs()[0]
    #     # memory_initial = gpu.memoryUsed

    #     for n_vars in range(1000, max_vars + 1, step_vars):
    #         for n_samples in range(10000, max_samples + 1, step_samples):
    #             try:
    #                 # Generate random DAG and data
    #                 np.random.seed(42)
    #                 # Create a random upper triangular matrix for DAG
    #                 dag = np.triu(np.random.binomial(1, 0.3, (n_vars, n_vars)), k=1)
                    
    #                 # Generate data using matrix multiplication
    #                 noise = np.random.normal(0, 0.1, (n_samples, n_vars))
    #                 coefs = np.random.uniform(0.5, 1.5, size=(n_vars, n_vars)) * dag
    #                 data = np.zeros((n_samples, n_vars))
                    
    #                 # Topological order for data generation
    #                 order = np.arange(n_vars)
    #                 for i in order:
    #                     data[:, i] = data @ coefs[:, i] + noise[:, i]
                    
    #                 df = pd.DataFrame(data)
                    
    #                 # Start memory monitoring thread
    #                 monitor_thread = threading.Thread(target=monitor_memory)
    #                 monitor_thread.daemon = True
    #                 monitor_thread.start()
                    
    #                 # Record start time
    #                 start_time = time.time()
                    
    #                 # Run PC algorithm
    #                 _, _, _ = self.fit(df)
                    
    #                 # Record end time
    #                 end_time = time.time()
                    
    #                 # Stop memory monitoring
    #                 monitor_thread.join(timeout=1)
                    
    #                 # Get peak memory usage
    #                 peak_mem = 0
    #                 while not memory_queue.empty():
    #                     mem = memory_queue.get()
    #                     peak_mem = max(peak_mem, mem) - memory_initial
                    
    #                 # Calculate execution time
    #                 exec_time = end_time - start_time
                    
    #                 print(f"{n_vars:5d} {n_samples:8d} {peak_mem:12.1f} {exec_time:10.2f} {'Success':>10}")
                    
    #                 # Store results
    #                 results.append({
    #                     'n_vars': n_vars,
    #                     'n_samples': n_samples,
    #                     'memory_used_mb': peak_mem,
    #                     'execution_time_s': exec_time,
    #                     'status': 'Success'
    #                 })
                    
    #                 # Check if memory usage is approaching GPU limit
    #                 gpu = GPUtil.getGPUs()[0]
    #                 if peak_mem > 0.9 * gpu.memoryTotal:
    #                     print("\nWarning: Approaching GPU memory limit!")
    #                     print(f"Test stopped at {n_vars} variables and {n_samples} samples")
                        
    #                     # Save results before returning
    #                     results_df = pd.DataFrame(results)
    #                     results_df.to_csv('stress_test_results.csv', index=False)
    #                     return
                        
    #             except Exception as e:
    #                 print(f"{n_vars:5d} {n_samples:8d} {'N/A':>12} {'N/A':>10} {'Failed':>10}")
    #                 print(f"Error: {str(e)}")
                    
    #                 # Store failed attempt
    #                 results.append({
    #                     'n_vars': n_vars,
    #                     'n_samples': n_samples,
    #                     'memory_used_mb': None,
    #                     'execution_time_s': None,
    #                     'status': f'Failed: {str(e)}'
    #                 })
                    
    #                 # Save results before returning
    #                 results_df = pd.DataFrame(results)
    #                 results_df.to_csv('stress_test_results.csv', index=False)
    #                 return
                    
    #     # Save final results if loop completes
    #     results_df = pd.DataFrame(results)
    #     results_df.to_csv('stress_test_results.csv', index=False)

if __name__ == "__main__":
    pc_algo = AcceleratedPC({})
    pc_algo.test_algorithm()