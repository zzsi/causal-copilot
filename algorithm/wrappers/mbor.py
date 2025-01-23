import numpy as np
import pandas as pd
from typing import Dict, Tuple

# Use local pyCausalFS package
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pycausalfs_dir = os.path.join(root_dir, 'externals', 'pyCausalFS')
if not os.path.exists(pycausalfs_dir):
    raise FileNotFoundError(f"Local pyCausalFS directory not found: {pycausalfs_dir}")
sys.path.append(pycausalfs_dir)
sys.path.append(root_dir)

from CBD.MBs.MBOR import MBOR as Mbor
from algorithm.wrappers.base import CausalDiscoveryAlgorithm

class MBOR(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 0.05,
            'is_discrete': True,
        }
        self._params.update(params)

    @property
    def name(self):
        return "MBOR"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['alpha', 'is_discrete']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        return {}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Fit the MBOR algorithm to data.
        
        Args:
            data: pandas DataFrame containing the data
            
        Returns:
            Tuple containing:
            - adj_matrix: numpy array representing the adjacency matrix with edge type 2 for MB
            - info: dictionary with additional information
        """
        n_vars = data.shape[1]
        adj_matrix = np.zeros((n_vars, n_vars))
        
        params = self.get_primary_params()
        total_ci_tests = 0
        
        # Run MBOR for each target variable
        for target in range(n_vars):
            mb, ci_num = Mbor(
                data=data,
                target=target,
                alpha=params['alpha'],
                is_discrete=params['is_discrete']
            )
            
            # Update adjacency matrix with Markov blanket using edge type 2 (undirected)
            for node in mb:
                adj_matrix[target, node] = 2
                
            total_ci_tests += ci_num

        info = {
            'total_ci_tests': total_ci_tests,
        }

        return adj_matrix, info

    def test_algorithm(self):
        """Run a simple test of the algorithm on synthetic data."""
        # Generate sample data
        np.random.seed(42)
        n_samples = 1000
        X1 = np.random.normal(0, 1, n_samples)
        X2 = 0.5 * X1 + np.random.normal(0, 0.5, n_samples)
        X3 = 0.3 * X1 + 0.7 * X2 + np.random.normal(0, 0.3, n_samples)
        X4 = 0.6 * X2 + np.random.normal(0, 0.4, n_samples)
        X5 = 0.4 * X3 + 0.5 * X4 + np.random.normal(0, 0.2, n_samples)
        
        df = pd.DataFrame({
            'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5
        })

        print("Testing MBOR algorithm with synthetic data:")
        adj_matrix, info = self.fit(df)
        
        print("\nAdjacency Matrix:")
        print(adj_matrix)
        print("\nAdditional Info:")
        print(f"Total CI tests: {info['total_ci_tests']}")

        # Ground truth adjacency matrix (using edge type 2 for undirected edges)
        gt_matrix = np.array([
            [0, 2, 2, 0, 0],
            [2, 0, 2, 2, 0],
            [2, 2, 0, 0, 2],
            [0, 2, 0, 0, 2],
            [0, 0, 2, 2, 0]
        ])

        # Compare with ground truth
        correct_edges = np.sum((adj_matrix == gt_matrix).astype(int))
        total_edges = np.sum(gt_matrix > 0)
        accuracy = correct_edges / (total_edges * 2)
        
        print(f"\nAccuracy: {accuracy:.4f}")

if __name__ == "__main__":
    mbor = MBORWrapper()
    mbor.test_algorithm() 