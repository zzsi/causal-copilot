import numpy as np
import pandas as pd
from typing import Dict, Tuple

import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.append(root_dir)
sys.path.append(algorithm_dir)

from tigramite.pcmci import PCMCI as PCMCI_model
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr

from algorithm.wrappers.base import CausalDiscoveryAlgorithm
from algorithm.evaluation.evaluator import GraphEvaluator
from causalnex.structure.data_generators import gen_stationary_dyn_net_and_df


class PCMCI(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'cond_ind_test': 'parcorr',
            'tau_min': 0,
            'tau_max': 1,
            'pc_alpha': 0.05,
            'alpha_level': 0.05,
            'fdr_method': 'none',
            'link_assumptions': None,
            'max_conds_dim': None,
            'max_combinations': 1,
            'max_conds_py': None,
            'max_conds_px': None,
        }
        self._params.update(params)

    @property
    def name(self):
        return "PCMCI"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['tau_min', 'tau_max', 'pc_alpha', 'alpha_level']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['link_assumptions', 'max_conds_dim', 'max_combinations', 'max_conds_py', 'max_conds_px']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, PCMCI_model]:
        # PCMCI
        node_names = list(data.columns)
        data_t = pp.DataFrame(data.values, var_names=node_names)
        
        if self._params['cond_ind_test'] == 'parcorr':
            cond_ind_test = ParCorr()
        elif self._params['cond_ind_test'] == 'robustparcorr':
            cond_ind_test = RobustParCorr()
        elif self._params['cond_ind_test'] == 'gpdc':
            from tigramite.independence_tests.gpdc import GPDC
            cond_ind_test = GPDC(significance='analytic', gp_params=None)
        elif self._params['cond_ind_test'] == 'gsq':
            from tigramite.independence_tests.gsquared import Gsquared
            cond_ind_test = Gsquared(significance='analytic')
        elif self._params['cond_ind_test'] == 'regression':
            from tigramite.independence_tests.regressionCI import RegressionCI
            cond_ind_test = RegressionCI(significance='analytic')
        elif self._params['cond_ind_test'] == 'cmi':
            from tigramite.independence_tests.cmiknn import CMIknn
            cond_ind_test = CMIknn(significance='shuffle_test', knn=0.1, shuffle_neighbors=5, transform='ranks', sig_samples=5)

        
        pcmci = PCMCI_model(dataframe=data_t, cond_ind_test=cond_ind_test)
        results = pcmci.run_pcmci(**self.get_primary_params(), **self.get_secondary_params())
        if self._params['fdr_method'] !='none':
            q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], 
                                                fdr_method=self._params['fdr_method'],
                                                exclude_contemporaneous=False)
        else:
            q_matrix = results['p_matrix']
        
        # Prepare additional information
        info = {
            'val_matrix': results['val_matrix'],
            'p_matrix': results['p_matrix'],
            'conf_matrix': results['conf_matrix'],
            'alpha': self._params['alpha_level'],
            'q_matrix': q_matrix
        }
        
        matrices = (q_matrix <= self._params['alpha_level']).astype(int)
        lag_matrix = np.array([matrices[:, :, lag].T for lag in range(matrices.shape[2])])
        
        return lag_matrix, info, results
    
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
    
    def generate_data(self, n=5, T=1000, random_state=None):
        '''
        n = 5  # Number of variables
        T = 1000  # Number of time steps
        random_state = None  # Random seed (None means no fixed seed)
        '''
        
        np.random.seed(random_state)  # Set the seed (not effective here since it's None)

        T_spurious = 20  # Extra time steps used to remove startup artifacts # This adds extra time steps at the beginning to stabilize the system before collecting actual data.
        expon = 1.5  # Exponent used for non-Gaussian noise # Used later to generate non-Gaussian noise.
        
        '''
        Used for randomly generating causal adjacent coefficient matrix
        # Constructing B0 (Instantaneous Effect Matrix): Generates a random matrix B0 that defines the instantaneous causal relationships between the 5 variables.
        value = np.random.uniform(low=0.05, high=0.5, size=(n, n))  # Random weights (0.05 to 0.5)
        sign = np.random.choice([-1, 1], size=(n, n))  # Randomly assign positive or negative sign
        B0 = np.multiply(value, sign)  # Multiply values by their signs
        B0 = np.multiply(B0, np.random.binomial(1, 0.4, size=(n, n)))  # Apply sparsity (40% chance of being nonzero)
        B0 = np.tril(B0, k=-1)  # Keep only lower triangular part (ensuring acyclic structure)
        
        # Constructing B1 (Lagged Effect Matrix): Similar to B0, but represents lagged dependencies (i.e., how past values influence current values).
        value = np.random.uniform(low=0.05, high=0.5, size=(n, n))
        sign = np.random.choice([-1, 1], size=(n, n))
        B1 = np.multiply(value, sign)
        B1 = np.multiply(B1, np.random.binomial(1, 0.4, size=(n, n)))
        # M1 = np.dot(np.linalg.inv(np.eye(n) - B0), B1)  # Computes the lagged influence matrix M1 by accounting for instantaneous effects (B0) using matrix inversion.
        '''
        # manually set adjacent coefficient matrix
        B0 = [
            [0,-0.12,0,0,0],
            [0,0,0,0,0],
            [-0.41,0.01,0,-0.02,0],
            [0.04,-0.22,0,0,0],
            [0.15,0,-0.03,0,0],
        ]
        B1 = [
            [-0.32,0,0.12,0.32,0],
            [0,-0.35,-0.1,-0.46,0.4],
            [0,0,0.37,0,0.46],
            [-0.38,-0.1,-0.24,0,-0.13],
            [0,0,0,0,0],
        ]
        causal_order = [1, 0, 3, 2, 4]
        
        # Generating Non-Gaussian Noise # Generates non-Gaussian noise by raising absolute values to the power of expon (1.5).
        ee = np.empty((n, T + T_spurious))
        for i in range(n):
            ee[i, :] = np.random.normal(size=(1, T + T_spurious))  # Standard normal noise
            ee[i, :] = np.multiply(np.sign(ee[i, :]), abs(ee[i, :]) ** expon)  # Introduce non-Gaussianity
            ee[i, :] = ee[i, :] - np.mean(ee[i, :])  # Normalize mean to 0
            ee[i, :] = ee[i, :] / np.std(ee[i, :])  # Normalize standard deviation to 1

        # Adjusting Noise Influence
        std_e = np.random.uniform(size=(n,)) + 0.5  # Random noise standard deviation (0.5 to 1.5)
        nn = np.dot(np.dot(np.linalg.inv(np.eye(n) - B0), np.diag(std_e)), ee) # Adjusted noise that incorporates instantaneous effects (B0).
        
        
        # Simulating the Time Series (xx)
        xx = np.zeros((n, T + T_spurious))  # Initialize data storage
        xx[:, 0] = np.random.normal(size=(n,))  # Random initial state

        for t in range(1, T + T_spurious):
            # xx[:, t] = np.dot(M1, xx[:, t - 1]) + nn[:, t]      #The new value xx[:, t] is computed as: A function of past values (M1 * xx[:, t - 1]),  Plus new noise (nn[:, t])
            lag_influence = np.dot(B1, xx[:, t - 1])
            xx[:, t] = lag_influence + np.dot(B0, lag_influence) + nn[:, t]
            
        # Extracting Final Dataset
        data = xx[:, T_spurious + 1 : T_spurious + T]  # The first T_spurious time steps are removed to ensure a stable system.
        # df = pd.DataFrame(data.T, columns=["x1", "x2", "x3", "x4", "x5"])
        df = pp.DataFrame(data.T)
        return df
           
    def test_algorithm_base(self):
        # Generate sample data with linear relationships    # np.random.seed(42)    # n_samples = 1000
        df = self.generate_data(n=5, T=1000, random_state=42)

        print("Testing PCMCI algorithm with dataframe:")
        params = {
            'dataframe': None,
            'cond_ind_test': ParCorr,
            'verbosity': 0,
            'selected_links': None,
            'link_assumptions': None,
            'tau_min': 0,
            'tau_max': 1,
            'save_iterations': False,
            'pc_alpha': 0.2,
            'max_conds_dim': None,
            'max_combinations': 1,
            'max_conds_py': None,
            'max_conds_px': None,
            'alpha_level': 0.05,
            'fdr_method': 'none',  
        }
        adj_matrix, info, pcmci = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)
        
        print("\nAdditional Info:")
        # print("  Test Statistic Values (val_matrix):")
        # print("    ", info['val_matrix'])
        
        # print("  P-Values (p_matrix):")
        # print("    ", info['p_matrix'])
        
        # print("  Confidence Intervals (conf_matrix):")
        # print("    ", info['conf_matrix'])
        pcmci.print_significant_links(p_matrix=info['p_matrix'],
                                         val_matrix=info['val_matrix'],
                                         alpha_level=0.05)
  
  
        # Ground truth graph
        gt_graph = np.array([
            [
                [0,-0.12,0,0,0],
                [0,0,0,0,0],
                [-0.41,0.01,0,-0.02,0],
                [0.04,-0.22,0,0,0],
                [0.15,0,-0.03,0,0],
            ],
            [
                [-0.32,0,0.12,0.32,0],
                [0,-0.35,-0.1,-0.46,0.4],
                [0,0,0.37,0,0.46],
                [-0.38,-0.1,-0.24,0,-0.13],
                [0,0,0,0,0],
            ]
        ])

    def test_algorithm(self):
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
        print("Testing algorithm with pandas DataFrame:")
        # Initialize lists to store metrics
        f1_scores = []
        precisions = []
        recalls = []
        shds = []
        
        for _ in range(2):
            adj_matrix,_,_ = self.fit(df)
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
        'cond_ind_test': 'robustparcorr',
        'tau_min': 0,
        'tau_max': 2,
        'pc_alpha': 1,
        'alpha_level': 0.07,
        'fdr_method': 'fdr_bh'
    }
    pcmci_algo = PCMCI(params)
    pcmci_algo.test_algorithm() 

