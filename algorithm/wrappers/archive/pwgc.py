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
from statsmodels.tsa.stattools import grangercausalitytests
from algorithm.wrappers.utils.ts_utils import generate_stationary_linear

# reference - https://github.com/ckassaad/causal_discovery_for_time_series

def granger_pw(X_train, p=5, alpha=0.05):
    test = 'ssr_ftest'
    names = X_train.columns
    adj = np.zeros((len(names), len(names)), dtype=int)
    dataset = pd.DataFrame(np.zeros((len(names), len(names)), dtype=int), columns=names, index=names)
    for c in dataset.columns:
        for r in dataset.index:
            test_result = grangercausalitytests(X_train[[r,c]], maxlag=p, verbose=False)
            p_values = [round(test_result[i+1][0][test][1], 4) for i in range(p)]
            min_p_value = np.min(p_values)
            # dataset.loc[c, r] = min_p_value
            if min_p_value < alpha:
                dataset.loc[c, r] = 2
                adj[int(r), int(c)] = 1
    for c in dataset.columns:
        for r in dataset.index:
            if dataset.loc[c, r] == 2:
                if dataset.loc[r, c] == 0:
                    dataset.loc[r, c] = 1
            if r == c:
                dataset.loc[r, c] = 1
                adj[int(c)][int(r)] = 1
    return dataset, adj


class PWGC(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'p': int,
            'alpha': 0.1, #significance level for F test
        }
        self._params.update(params)

    @property
    def name(self):
        return "PWGC"

    def get_params(self):
        return self._params
    
    def get_primary_params(self):
        self._primary_param_keys = ['p', 'alpha']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}
    
    def get_secondary_params(self):
        self._secondary_param_keys = []
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        node_names = list(data.columns)
        max_lag = self._params['p']
        res_df, summary_matrix = granger_pw(data, **self.get_primary_params())
        
        info = {
            'lag': max_lag,
            'nodes': node_names,
        }

        return summary_matrix, info, res_df

    def test_algorithm(self):
        # Generate some sample data
        np.random.seed(42)
        n_samples = 1000
        n_nodes = 3
        lag = 2
        
        df, gt_graph_lag, gt_graph_summary, graph_net = generate_stationary_linear(
            n_nodes,
            n_samples,
            lag,
            degree_intra=1,
            degree_inter=2,
        )
        print("Testing MVGC algorithm with pandas DataFrame:")
        print("Ground truth summary graph\n", gt_graph_summary)
        # Initialize lists to store metrics
        f1_scores = []
        precisions = []
        recalls = []
        shds = []
        
        # Run the algorithm
        for _ in range(2):
            prediction, _, _ = self.fit(df)
            print("Prediction\n", prediction)
            evaluator = GraphEvaluator()
            metrics = evaluator._compute_single_metrics(gt_graph_summary, prediction)
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
        'p': 2,
        'alpha': 0.5,
    }
    pwgc_algo = PWGC(params)
    pwgc_algo.test_algorithm() 