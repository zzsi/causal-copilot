import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, List

import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.append(root_dir)
sys.path.append(algorithm_dir)

from algorithm.wrappers.base import CausalDiscoveryAlgorithm
from algorithm.evaluation.evaluator import GraphEvaluator
from castle.algorithms import GOLEM as golem

class GOLEM(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'lambda1': 2e-2,  # L1 penalty coefficient
            'lambda2': 5.0,  # DAG penalty coefficient
            'equal_variances': True,  # Whether to assume equal noise variances
            'non_equal_variances': True,  # Whether to assume non-equal noise variances
            'learning_rate': 1e-3,  # Learning rate for Adam optimizer
            'max_iter': 1e5,  # Number of training iterations
            'checkpoint_iter': 5000,  # Iterations between checkpoints
            'seed': 1,  # Random seed
            'graph_thres': 0.3,  # Threshold for weighted matrix
            'device_type': 'auto',  # Device type ('cpu' or 'gpu' or 'auto')
            'device_ids': None  # GPU device IDs to use
        }
        self._params.update(params)
        # Automatically decide device_type if set to 'auto'
        if self._params.get('device_type', 'cpu') == 'auto':
            try:
                import torch
                self._params['device_type'] = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self._params['device_type'] = 'cpu'

    @property
    def name(self):
        return "GOLEM"

    def get_params(self):
        return self._params
    
    def get_primary_params(self):
        self._primary_param_keys = ['lambda1', 'lambda2', 'max_iter']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}
    
    def get_secondary_params(self):
        self._secondary_param_keys = ['learning_rate', 'equal_variances', 'non_equal_variances', 
                                      'checkpoint_iter', 'seed', 'graph_thres', 'device_type', 'device_ids']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        if isinstance(data, pd.DataFrame):
            node_names = list(data.columns)
            data = data.values
        else:
            node_names = [f"X{i}" for i in range(data.shape[1])]

        model = golem(lambda_1=self._params['lambda1'],
                     lambda_2=self._params['lambda2'],
                     equal_variances=self._params['equal_variances'],
                     non_equal_variances=self._params['non_equal_variances'],
                     learning_rate=self._params['learning_rate'],
                     num_iter=self._params['max_iter'],
                     checkpoint_iter=self._params['checkpoint_iter'],
                     seed=self._params['seed'],
                     graph_thres=self._params['graph_thres'],
                     device_type=self._params['device_type'],
                     device_ids=self._params['device_ids'])

        model.learn(data)
        
        # GOLEM returns transposed matrix compared to our convention
        adj_matrix = model.causal_matrix

        info = {
            'model': model,
            'node_names': node_names
        }

        return adj_matrix, info

    def test_algorithm(self):
        # Generate sample data
        np.random.seed(42)
        n_samples = 1000
        X1 = np.random.normal(0, 1, n_samples)
        X2 = 0.5 * X1 + np.random.normal(0, 0.5, n_samples)
        X3 = 0.3 * X1 + 0.7 * X2 + np.random.normal(0, 0.3, n_samples)
        X4 = 0.6 * X2 + np.random.normal(0, 0.4, n_samples)
        X5 = 0.4 * X3 + 0.5 * X4 + np.random.normal(0, 0.2, n_samples)
        
        df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5})

        print("Testing GOLEM algorithm with pandas DataFrame:")

        # Ground truth graph
        gt_graph = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0]
        ])

        # Initialize lists to store metrics
        f1_scores = []
        precisions = []
        recalls = []
        shds = []

        # Run the algorithm 10 times
        for _ in range(10):
            adj_matrix, info = self.fit(df)
            evaluator = GraphEvaluator()
            metrics = evaluator.compute_metrics(gt_graph, adj_matrix)
            f1_scores.append(metrics['f1'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            shds.append(metrics['shd'])

        # Calculate average and standard deviation
        avg_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        avg_precision = np.mean(precisions)
        std_precision = np.std(precisions)
        avg_recall = np.mean(recalls)
        std_recall = np.std(recalls)
        avg_shd = np.mean(shds)
        std_shd = np.std(shds)

        print("\nAverage Metrics over 10 runs:")
        print(f"F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
        print(f"Precision: {avg_precision:.4f} ± {std_precision:.4f}")
        print(f"Recall: {avg_recall:.4f} ± {std_recall:.4f}")
        print(f"SHD: {avg_shd:.4f} ± {std_shd:.4f}")

if __name__ == "__main__":
    golem_algo = GOLEM({})
    golem_algo.test_algorithm()
