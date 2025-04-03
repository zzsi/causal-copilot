import numpy as np
import pandas as pd
from typing import Dict, Tuple

# use the local lingam package
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
causal_learn_dir = os.path.join(root_dir, 'externals', 'causal-learn')
if not os.path.exists(causal_learn_dir):
    raise FileNotFoundError(f"Local causal-learn directory not found: {causal_learn_dir}, please git clone the submodule of causal-learn")
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.append(root_dir)
sys.path.append(algorithm_dir)
sys.path.append(causal_learn_dir)

from causallearn.search.FCMBased.lingam import VARLiNGAM as VARLiNGAM_model

from algorithm.wrappers.base import CausalDiscoveryAlgorithm
from algorithm.evaluation.evaluator import GraphEvaluator
from algorithm.wrappers.utils.ts_utils import generate_stationary_linear

import torch
cuda_available = torch.cuda.is_available()
try:
    from culingam.varlingam import VarLiNGAM as AcVarLiNGAM
except ImportError:
    if not cuda_available:
        print("CUDA is not available, will not use GPU acceleration")

# {‘aic’, ‘fpe’, ‘hqic’, ‘bic’, None}
class VARLiNGAM(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'lags': 1,
            'criterion': "bic",
            'prune': True,
            'ar_coefs': None,
            'lingam_model': None,
            'random_state': None,
            'gpu': False
        }
        self._params.update(params)

    @property
    def name(self):
        return "VARLiNGAM"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['lags', 'criterion', 'prune', 'gpu']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['ar_coefs', 'lingam_model', 'random_state']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}


    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, VARLiNGAM_model]:
        # Check and remove domain_index if it exists
        if 'domain_index' in data.columns:
            data = data.drop(columns=['domain_index'])
            
        node_names = list(data.columns)
        data_values = data.values

        # Combine primary and secondary parameters
        all_params = {**self.get_primary_params(), **self.get_secondary_params()}
        all_params.pop('gpu')

        if cuda_available and self._params['gpu']:
            model = AcVarLiNGAM(**all_params)
        else:
            model = VARLiNGAM_model(**all_params)
        model.fit(data_values)
        
        lag_matrix = model._adjacency_matrices

        # Prepare additional information
        info = {
            'model': model,
            'lag': model._lags,
            'residuals': model._residuals,
            'causal_order': model.causal_order_,
            'lag_matrix': lag_matrix
        }
        summary_matrix = np.any(lag_matrix, axis=0).astype(int)
        
        return summary_matrix, info, model
    
    def test_algorithm(self):
        # Generate some sample data
        np.random.seed(42)
        n_samples = 1000
        n_nodes = 3
        lag = 2
        
        df, gt_graph, gt_summary, graph_net = generate_stationary_linear(
            n_nodes,
            n_samples,
            lag,
            degree_intra=1,
            degree_inter=2,
        )
        print("Testing VARLINGAM algorithm with pandas DataFrame:")
        print("Ground truth graph\n", gt_graph)
        # Initialize lists to store metrics
        f1_scores = []
        precisions = []
        recalls = []
        shds = []
        
        # Run the algorithm
        for _ in range(1):
            adj_matrix, _, _ = self.fit(df)
            print("Prediction\n", adj_matrix)
            evaluator = GraphEvaluator()
            metrics = evaluator._compute_single_metrics(gt_summary, adj_matrix)
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

        print("\nAverage Metrics:")
        print(f"F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
        print(f"Precision: {avg_precision:.4f} ± {std_precision:.4f}")
        print(f"Recall: {avg_recall:.4f} ± {std_recall:.4f}")
        print(f"SHD: {avg_shd:.4f} ± {std_shd:.4f}")

if __name__ == "__main__":
    params = {
        'lags': 2,
        'criterion': "bic",
        'prune': False,
        'ar_coefs': None,
        'lingam_model': None,
        'random_state': None,
    }
    algo = VARLiNGAM(params)
    algo.test_algorithm() 

