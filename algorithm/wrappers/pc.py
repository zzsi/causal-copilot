import numpy as np
import pandas as pd
from typing import Dict, Tuple
import json
import os

# use the local causal-learn package
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
causal_learn_dir = os.path.join(root_dir, 'externals', 'causal-learn')
if not os.path.exists(causal_learn_dir):
    raise FileNotFoundError(f"Local causal-learn directory not found: {causal_learn_dir}, please git clone the submodule of causal-learn")
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.append(root_dir)
sys.path.append(algorithm_dir)
sys.path.insert(0, causal_learn_dir)

from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.ConstraintBased.PC import pc as cl_pc
from algorithm.wrappers.base import CausalDiscoveryAlgorithm
from algorithm.evaluation.evaluator import GraphEvaluator

import torch
cuda_available = torch.cuda.is_available()
try:
    from externals.acceleration.pc.pc import accelerated_pc
except ImportError:
    if not cuda_available:
        raise ImportError("CUDA is not available, will not use GPU acceleration")


# KCI: 5 nodes, 1000 samples: 66s
# KCI: 5 nodes, 3000 samples: 100s
# kci: 10 nodes, 1000 samples: 500s
# KCI: 20 nodes, 1000 samples: 5500s
# fastkci: 5 nodes, 1000 samples: 15s
# fastkci: 5 nodes, 3000 samples: 80s
# fastkci: 10 nodes, 1000 samples: 160s
# fastkci: 20 nodes, 1000 samples: 1000s
# rcit: 5 nodes, 1000 samples: 2s
# rcit: 5 nodes, 3000 samples: 3s
# rcit: 5 nodes, 10000 samples: 25s
# rcit: 10 nodes, 1000 samples: 17s
# rcit: 20 nodes, 1000 samples: 180s


class PC(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 0.05,
            'indep_test': 'fisherz_cpu',  # Default to fisherz
            'depth': 3,
            'stable': True,
            'uc_rule': 0,
            'uc_priority': -1,
            'mvpc': False,
            'correction_name': 'MV_Crtn_Fisher_Z',
            'background_knowledge': None,
            'verbose': False,
            'show_progress': False,
        }
        self._params.update(params)

    @property
    def name(self):
        return "PC"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['alpha', 'indep_test', 'depth']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['stable', 'uc_rule', 'uc_priority', 'mvpc', 'correction_name',
                                    'background_knowledge', 'verbose', 'show_progress', 'gamma']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, CausalGraph]:
        node_names = list(data.columns)
        data_values = data.values

        if cuda_available and 'gpu' in self._params['indep_test']:
            use_gpu = True
            self._params['indep_test'] = self._params['indep_test'].replace('_gpu', '')
        else:
            use_gpu = False
            self._params['indep_test'] = self._params['indep_test'].replace('_cpu', '')
        
        if use_gpu:
            # Use GPU implementation
            all_params = {
                'alpha': self._params['alpha'],
                'indep_test': self._params['indep_test'],
                'depth': self._params['depth'],
            }
            adj_matrix, info = accelerated_pc(data_values, **all_params)
            adj_matrix = self.convert_to_adjacency_matrix(adj_matrix)
            cg = adj_matrix
        else:
            # Use CPU implementation
            all_params = {**self.get_primary_params(), **self.get_secondary_params(), 'node_names': node_names}
            cg = cl_pc(data_values, **all_params)
            adj_matrix = self.convert_to_adjacency_matrix(cg.G.graph)

            # Prepare additional information
            info = {
                'sepset': cg.sepset if hasattr(cg, 'sepset') else None,
                'definite_UC': cg.definite_UC if hasattr(cg, 'definite_UC') else [],
                'definite_non_UC': cg.definite_non_UC if hasattr(cg, 'definite_non_UC') else [],
            }

        return adj_matrix, info, cg
    
    def convert_to_adjacency_matrix(self, adj_matrix: np.ndarray) -> np.ndarray:
        # Handle both GPU and CPU graph formats
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
        import time
        start_time = time.time()
        np.random.seed(42)
        n_samples = 1000
        X1 = np.random.normal(0, 1, n_samples)
        X2 = 0.5 * X1 + np.random.normal(0, 0.5, n_samples)
        X3 = 0.3 * X1 + 0.7 * X2 + np.random.normal(0, 0.3, n_samples)
        X4 = 0.6 * X2 + np.random.normal(0, 0.4, n_samples)
        X5 = 0.4 * X3 + 0.5 * X4 + np.random.normal(0, 0.2, n_samples)
        X6 = 0.3 * X4 + 0.2 * X5 + np.random.normal(0, 0.3, n_samples)
        X7 = 0.5 * X1 + 0.3 * X6 + np.random.normal(0, 0.4, n_samples)
        X8 = 0.6 * X3 + 0.4 * X7 + np.random.normal(0, 0.3, n_samples)
        X9 = 0.2 * X6 + 0.7 * X8 + np.random.normal(0, 0.25, n_samples)
        X10 = 0.8 * X9 + 0.1 * X5 + np.random.normal(0, 0.2, n_samples)
        # X11 = 0.4 * X10 + 0.3 * X7 + np.random.normal(0, 0.3, n_samples)
        # X12 = 0.5 * X8 + 0.2 * X11 + np.random.normal(0, 0.35, n_samples)
        # X13 = 0.7 * X5 + 0.3 * X9 + np.random.normal(0, 0.25, n_samples)
        # X14 = 0.6 * X13 + 0.2 * X11 + np.random.normal(0, 0.3, n_samples)
        # X15 = 0.4 * X12 + 0.5 * X14 + np.random.normal(0, 0.2, n_samples)
        # X16 = 0.3 * X10 + 0.6 * X15 + np.random.normal(0, 0.25, n_samples)
        # X17 = 0.5 * X14 + 0.3 * X16 + np.random.normal(0, 0.3, n_samples)
        # X18 = 0.4 * X16 + 0.2 * X13 + np.random.normal(0, 0.35, n_samples)
        # X19 = 0.6 * X17 + 0.3 * X18 + np.random.normal(0, 0.25, n_samples)
        # X20 = 0.5 * X18 + 0.4 * X19 + np.random.normal(0, 0.2, n_samples)

        # Ground truth graph (20×20 adjacency matrix)
        gt_graph = np.zeros((5, 5))
        # X1 -> X2, X3, X7
        gt_graph[1, 0] = 1
        gt_graph[2, 0] = 1
        # gt_graph[6, 0] = 1
        # X2 -> X3, X4
        gt_graph[2, 1] = 1
        gt_graph[3, 1] = 1
        # X3 -> X5, X8
        gt_graph[4, 2] = 1
        # gt_graph[7, 2] = 1
        # X4 -> X5, X6
        gt_graph[4, 3] = 1
        # gt_graph[5, 3] = 1
        # # X5 -> X6, X13
        # gt_graph[5, 4] = 1
        # # gt_graph[12, 4] = 1
        # # X6 -> X7, X9
        # gt_graph[6, 5] = 1
        # gt_graph[8, 5] = 1
        # # X7 -> X8, X11
        # gt_graph[7, 6] = 1
        # # gt_graph[10, 6] = 1
        # # X8 -> X9, X12
        # gt_graph[8, 7] = 1
        # # gt_graph[11, 7] = 1
        # # X9 -> X10, X13
        # gt_graph[9, 8] = 1
        # gt_graph[12, 8] = 1
        # X10 -> X11, X16
        # gt_graph[10, 9] = 1
        # # gt_graph[15, 9] = 1
        # # X11 -> X12, X14
        # gt_graph[11, 10] = 1
        # gt_graph[13, 10] = 1
        # # X12 -> X15
        # gt_graph[14, 11] = 1
        # # X13 -> X14, X18
        # gt_graph[13, 12] = 1
        # gt_graph[17, 12] = 1
        # # X14 -> X15, X17
        # gt_graph[14, 13] = 1
        # gt_graph[16, 13] = 1
        # # X15 -> X16
        # gt_graph[15, 14] = 1
        # # X16 -> X17, X18
        # gt_graph[16, 15] = 1
        # gt_graph[17, 15] = 1
        # # X17 -> X19
        # gt_graph[18, 16] = 1
        # # X18 -> X19, X20
        # gt_graph[18, 17] = 1
        # gt_graph[19, 17] = 1
        # # X19 -> X20
        # gt_graph[19, 18] = 1
        
        df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5}) #, 'X6': X6, 'X7': X7, 'X8': X8, 'X9': X9, 'X10': X10})

        print("Testing PC algorithm with pandas DataFrame:")

        adj_matrix, info, _ = self.fit(df)
        end_time = time.time()
        print(f"PC elapsed time: {end_time - start_time:.4f} seconds")
        print("Adjacency Matrix:")
        print(adj_matrix)
        print("\nAdditional Info:")
        # print(f"PC elapsed time: {info['PC_elapsed']:.4f} seconds")
        # print(f"Number of definite unshielded colliders: {len(info['definite_UC'])}")
        # print(f"Number of definite non-unshielded colliders: {len(info['definite_non_UC'])}")


        # Use GraphEvaluator to compute metrics
        evaluator = GraphEvaluator()
        metrics = evaluator.compute_metrics(gt_graph, adj_matrix)

        print("\nMetrics:")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"SHD: {metrics['shd']:.4f}")

if __name__ == "__main__":
    # for indep_test in ['rcit_cpu', 'cmiknn_gpu']:
    #     print(f"Testing {indep_test}")
    pc_algo = PC()
    pc_algo.test_algorithm() 