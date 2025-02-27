import numpy as np
import pandas as pd
from typing import Dict, Tuple

# use the local lingam package
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.append(root_dir)
sys.path.append(algorithm_dir)

from lingam import VARLiNGAM as VARLiNGAM_model
from causalnex.structure.data_generators import gen_stationary_dyn_net_and_df


from algorithm.wrappers.base import CausalDiscoveryAlgorithm
from algorithm.evaluation.evaluator import GraphEvaluator

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
        }
        self._params.update(params)

    @property
    def name(self):
        return "VARLiNGAM"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['lags', 'criterion', 'prune']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['ar_coefs', 'lingam_model', 'random_state']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}


    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, VARLiNGAM_model]:
        node_names = list(data.columns)
        data_values = data.values

        # Combine primary and secondary parameters
        all_params = {**self.get_primary_params(), **self.get_secondary_params()}

        model = VARLiNGAM_model(**all_params)
        model.fit(data_values)
        
        adj_matrices = model._adjacency_matrices

        # Prepare additional information
        info = {
            'coefficients': model._ar_coefs,
            'number_of_lags': model._lags,
            'residuals': model._residuals,
            'causal_order': model.causal_order_,
        }

        return adj_matrices, info, model
    
    
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
        df = pd.DataFrame(data.T, columns=["x1", "x2", "x3", "x4", "x5"])
        
        return df
 
    def test_algorithm2(self):
        # Generate sample data with linear relationships    # np.random.seed(42)    # n_samples = 1000
        df = self.generate_data(n=5, T=1000, random_state=42)

        print("Testing VAR-LiNGAM algorithm with pandas:")
        params = {
            'lags': 1,
            'criterion': "bic",
            'prune': True,
            'ar_coefs': None,
            'lingam_model': None,
            'random_state': None,
        }
        adj_matrix, info, _ = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)
        
        print("\nAdditional Info:")
        # # Printing coefficients # self._ar_coefs = M_taus
        # print("\nCoefficients (AR coefficients):")
        # for i, coef in enumerate(info['coefficients']):
        #     print(f"Lag {i+1} Coefficients:\n{coef}")
        # Printing number of lags
        print(f"  Number of lags: {info['number_of_lags']}")
        # Printing residuals summary (assuming residuals are a NumPy array)
        print("  Residuals Summary:")
        print(f"    Mean: {np.mean(info['residuals'], axis=0)}")
        print(f"    Std Dev: {np.std(info['residuals'], axis=0)}")
        # Printing causal order
        print("  Causal Order:")
        print("    "," -> ".join(map(str, info['causal_order'])))
        
        
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

        # need to update GraphEvaluator() to allow multiple adjacent matrices
        # # Use GraphEvaluator to compute metrics
        # evaluator = GraphEvaluator()
        # metrics = evaluator.compute_metrics(gt_graph, adj_matrix)

        # print("\nMetrics:")
        # print(f"F1 Score: {metrics['f1']:.4f}")
        # print(f"Precision: {metrics['precision']:.4f}")
        # print(f"Recall: {metrics['recall']:.4f}")
        # print(f"SHD: {metrics['shd']:.4f}")

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
        'lags': 2,
        'criterion': "bic",
        'prune': False,
        'ar_coefs': None,
        'lingam_model': None,
        'random_state': None,
    }
    algo = VARLiNGAM(params)
    algo.test_algorithm() 

