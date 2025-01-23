import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, List

import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from algorithm.wrappers.base import CausalDiscoveryAlgorithm
from algorithm.evaluation.evaluator import GraphEvaluator
from externals.trustworthyAI.gcastle.castle.algorithms import NotearsNonlinear

class NOTEARSNonlinear(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'lambda1': 0.01,  # L1 regularization parameter
            'lambda2': 0.01,  # L2 regularization parameter
            'max_iter': 100,
            'h_tol': 1e-8,
            'rho_max': 1e+16,
            'w_threshold': 0.3,
            'hidden_layers': (10, 1),  # Default architecture with one hidden layer
            'bias': True,
            'model_type': 'mlp',  # 'mlp' or 'sob'
            'device_type': 'cpu',
            'device_ids': None,
            # Additional parameters for Sobolev model
            'expansions': 10  # Only used when model_type='sob'
        }
        self._params.update(params)

    @property
    def name(self):
        return "NOTEARS-MLP"

    def get_params(self):
        return self._params
    
    def get_primary_params(self):
        self._primary_param_keys = ['lambda1', 'lambda2', 'max_iter', 'hidden_layers', 
                                  'model_type', 'device_type']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}
    
    def get_secondary_params(self):
        self._secondary_param_keys = ['h_tol', 'rho_max', 'w_threshold', 'bias', 
                                    'device_ids', 'expansions']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        """
        Fit the NOTEARS-MLP algorithm to data.
        
        Args:
            data: Data matrix as numpy array or pandas DataFrame
            
        Returns:
            adj_matrix: Estimated adjacency matrix
            info: Additional information from the algorithm
            model: Fitted model object
        """
        if isinstance(data, pd.DataFrame):
            node_names = list(data.columns)
            data = data.values
        else:
            node_names = [f"X{i}" for i in range(data.shape[1])]
            data = np.array(data)

        # Initialize NOTEARS-MLP from Castle
        model = NotearsNonlinear(
            lambda1=self._params['lambda1'],
            lambda2=self._params['lambda2'],
            max_iter=self._params['max_iter'],
            h_tol=self._params['h_tol'],
            rho_max=self._params['rho_max'],
            w_threshold=self._params['w_threshold'],
            hidden_layers=self._params['hidden_layers'],
            expansions=self._params['expansions'],
            bias=self._params['bias'],
            model_type=self._params['model_type'],
            device_type=self._params['device_type'],
            device_ids=self._params['device_ids']
        )
        
        # Fit the model
        model.learn(data)
        
        # Get the adjacency matrix
        adj_matrix = model.causal_matrix
        if isinstance(adj_matrix, pd.DataFrame):
            adj_matrix = adj_matrix.values

        # Prepare additional information
        info = {
            'model': model,
            'weight_causal_matrix': model.weight_causal_matrix
        }

        return adj_matrix, info, model

    def test_algorithm(self):
        # Generate some sample data
        np.random.seed(42)
        n_samples = 1000
        X1 = np.random.normal(0, 1, n_samples)
        X2 = 0.5 * X1 + np.random.normal(0, 0.5, n_samples)
        X3 = 0.3 * X1 + 0.7 * X2 + np.random.normal(0, 0.3, n_samples)
        X4 = 0.6 * X2 + np.random.normal(0, 0.4, n_samples)
        X5 = 0.4 * X3 + 0.5 * X4 + np.random.normal(0, 0.2, n_samples)
        
        df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5})

        print("Testing NOTEARS-MLP algorithm with pandas DataFrame:")
        params = {
            'lambda1': 0.01,
            'lambda2': 0.01,
            'max_iter': 100,
            'h_tol': 1e-8,
            'w_threshold': 0.3,
            'hidden_layers': (10, 1),
            'model_type': 'mlp',
            'device_type': 'cpu'
        }
        adj_matrix, info, _ = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)

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

if __name__ == "__main__":
    notears_mlp = NOTEARSNonlinear()
    notears_mlp.test_algorithm()
