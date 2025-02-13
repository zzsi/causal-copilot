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
from causalnex.structure.dynotears import from_pandas_dynamic
from causalnex.structure.data_generators import gen_stationary_dyn_net_and_df

class DYNOTEARS(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'p': int,
            'lambda_w': 0.1,
            'lambda_a': 0.1,
            'max_iter': 100,
            'h_tol': 1e-8,
            'w_threshold': 0.01
        }
        self._params.update(params)

    @property
    def name(self):
        return "DYNOTEARS"

    def get_params(self):
        return self._params
    
    def get_primary_params(self):
        self._primary_param_keys = ['p', 'lambda_w', 'lambda_a', 'max_iter', 'w_threshold']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}
    
    def get_secondary_params(self):
        self._secondary_param_keys = ['h_tol']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        node_names = list(data.columns)
        data_values = data.values
        max_lag = self._params['p']
        #init graph_dict for result aggregation
        graph_dict = dict()
        for name in node_names:
            graph_dict[name] = []

        # call DYNOTEARS
        sm = from_pandas_dynamic(data, **self.get_primary_params(), **self.get_secondary_params())
        tname_to_name_dict = dict()
        count_lag = 0
        idx_name = 0
        for tname in sm.nodes:
            tname_to_name_dict[tname] = data.columns[idx_name]
            if count_lag == max_lag:
                idx_name = idx_name +1
                count_lag = -1
            count_lag = count_lag +1
            
        for ce in sm.edges:
            c = ce[0]
            e = ce[1]
            tc = int(c.partition("lag")[2])
            te = int(e.partition("lag")[2])
            t = tc - te
            if (tname_to_name_dict[c], -t) not in graph_dict[tname_to_name_dict[e]]:
                graph_dict[tname_to_name_dict[e]].append((tname_to_name_dict[c], -t))
                
        summary_matrix, lag_matrix = self.dict_to_adjacency_matrix(graph_dict, len(node_names), max_lag)
        
        return summary_matrix, lag_matrix
        
        
    def dict_to_adjacency_matrix(self, result_dict, num_nodes, lookback_period):
        # adj_matrix = np.zeros((lookback_period, num_nodes, num_nodes))
        lagged_adj_matrix = np.zeros((lookback_period + 1, num_nodes, num_nodes))
        
        nodes = list(result_dict.keys())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        for target, causes in result_dict.items():
            target_idx = node_to_idx[target]
            for cause, lag in causes:
                cause_idx = node_to_idx[cause]
                lag_index = -lag
                # if 0 <= lag_index <= lookback_period:
                # adj_matrix[lag_index, target_idx, cause_idx] = 1
                lagged_adj_matrix[lag_index, target_idx, cause_idx] = 1
                    
        summary_adj_matrix = np.any(lagged_adj_matrix, axis=0).astype(int)

        return summary_adj_matrix, lagged_adj_matrix

    def get_graph(self, sm, data):
        graph_dict = dict()
        for name in data:
            node = int(name.split('_')[0])
            graph_dict[node] = []

        for c, e in sm.edges:
            node_e = int(e.split('_')[0])
            node_c = int(c.split('_')[0])
            tc = int(c.partition("lag")[2])
            te = int(e.partition("lag")[2])
            if te !=0:
                print(c, e)
            lag = -1 * tc
            graph_dict[node_e].append((node_c, lag))
                
        return graph_dict

    def test_algorithm(self):
        # Generate some sample data

        np.random.seed(42)
        n_samples = 1000
        n_nodes = 3
        lag = 2
        graph_net, df, intra_nodes, inter_nodes = gen_stationary_dyn_net_and_df(
            num_nodes=n_nodes,
            n_samples=n_samples,
            p=lag,
            degree_intra=1,
            degree_inter=2,
            w_min_intra=0.04,
            w_max_intra=0.2,
            w_min_inter=0.06,
            w_max_inter=0.3,
            w_decay=1.0,
            sem_type='linear-gauss'
        )
        print("Sample data generated", inter_nodes, intra_nodes)
        graph_true = self.get_graph(graph_net, intra_nodes)
        summary, adj = self.dict_to_adjacency_matrix(graph_true, n_nodes, lag)
        gt_graph = np.column_stack(adj)
        print(gt_graph)
        df = df[intra_nodes]
        df.columns = [el.split('_')[0] for el in df.columns]
        print("Testing DYNOTEARS algorithm with pandas DataFrame:")
        
        # Initialize lists to store metrics
        f1_scores = []
        precisions = []
        recalls = []
        shds = []
        
        # Run the algorithm
        for _ in range(2):
            _, adj_matrix = self.fit(df)
            print(np.column_stack(adj_matrix))
            evaluator = GraphEvaluator()
            metrics = evaluator._compute_single_metrics(gt_graph, np.column_stack(adj_matrix))
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
        'lambda_w': 0.1,
        'lambda_a': 0.07,
        'max_iter': 1000,
        'w_threshold':0.01
    }
    dynotears_algo = DYNOTEARS(params)
    dynotears_algo.test_algorithm() 