import numpy as np
import pandas as pd
from typing import Dict, Tuple

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
sys.path.append(causal_learn_dir)

from externals.acceleration.cdnod.cdnod import accelerated_cdnod
from algorithm.wrappers.base import CausalDiscoveryAlgorithm
from algorithm.evaluation.evaluator import GraphEvaluator
from causallearn.graph.GraphClass import CausalGraph, GeneralGraph
from causallearn.utils.PCUtils import SkeletonDiscovery, UCSepset, Meek

class AcceleratedCDNOD(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 0.05,
            'indep_test': 'fisherz',
            'uc_rule': 0,
            'uc_priority': 2,
            'depth': -1,
        }
        self._params.update(params)

    @property
    def name(self):
        return "AcceleratedCDNOD"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['alpha', 'indep_test', 'depth']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        secondary_param_keys = ['uc_rule', 'uc_priority']
        return {k: v for k, v in self._params.items() if k in secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, CausalGraph]:
        """
        Runs GPU-accelerated CDNOD, ensuring c_index is correctly handled.
        """
        node_names = list(data.columns)
        data_values = data.values

        # **Extract context index (c_indx)**
        if 'domain_index' not in data.columns:
            raise ValueError("Dataset must contain a 'domain_index' column for CDNOD.")
        c_indx = data['domain_index'].values.reshape(-1, 1)

        # **Append c_index to data before passing it to CDNOD**
        data_aug = np.concatenate((data_values, c_indx), axis=1)

        # **Combine parameters**
        all_params = {**self.get_primary_params(), **self.get_secondary_params()}

        # **Run GPU-Accelerated CDNOD**
        cg = accelerated_cdnod(data_aug, c_indx, **all_params)

        # **Convert graph to adjacency matrix**
        adj_matrix = self.convert_to_adjacency_matrix(cg)

        info = {
            'graph': cg,
            'CDNOD_elapsed': cg.CDNOD_elapsed,
        }

        return adj_matrix, info, cg

    def convert_to_adjacency_matrix(self, cg: CausalGraph) -> np.ndarray:
        adj_matrix = cg.G.graph
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

    def apply_orientation(self, cg: CausalGraph, c_indx: np.ndarray) -> CausalGraph:
        """
        Apply UCSepset & Meek rules to orient edges in CDNOD.
        """
        c_indx_id = c_indx.shape[1] - 1

        # **Ensure `c_index` has directed edges to all adjacent nodes**
        for i in cg.G.get_adjacent_nodes(cg.G.nodes[c_indx_id]):
            cg.G.add_directed_edge(cg.G.nodes[c_indx_id], i)

        uc_rule = self._params['uc_rule']
        uc_priority = self._params['uc_priority']

        if uc_rule == 0:
            cg = UCSepset.uc_sepset(cg, uc_priority)
        elif uc_rule == 1:
            cg = UCSepset.maxp(cg, uc_priority)
        elif uc_rule == 2:
            cg = Meek.meek(UCSepset.definite_maxp(cg, self._params['alpha'], uc_priority))
        else:
            raise ValueError("uc_rule should be in [0, 1, 2]")

        return cg

    def test_algorithm(self):
        np.random.seed(42)
        n_samples = 1000
        X1 = np.random.normal(0, 1, n_samples)
        X2 = 0.5 * X1 + np.random.normal(0, 0.5, n_samples)
        X3 = 0.3 * X1 + 0.7 * X2 + np.random.normal(0, 0.3, n_samples)
        X4 = 0.6 * X2 + np.random.normal(0, 0.4, n_samples)
        X5 = 0.4 * X3 + 0.5 * X4 + np.random.normal(0, 0.2, n_samples)
        domain_index = np.ones_like(X1)

        df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 'domain_index': domain_index})

        print("Testing Accelerated CDNOD with pandas DataFrame:")
        params = {
            'alpha': 0.05,
            'depth': 2,
            'uc_rule': 1,
            'uc_priority': 2,
            'verbose': False
        }
        adj_matrix, info, _ = self.fit(df)

        print("Adjacency Matrix:")
        print(adj_matrix)
        print(f"CDNOD elapsed time: {info['CDNOD_elapsed']:.4f} seconds")

        evaluator = GraphEvaluator()
        metrics = evaluator.compute_metrics(adj_matrix[:-1, :-1], adj_matrix[:-1, :-1])

        print("\nMetrics:")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"SHD: {metrics['shd']:.4f}")


if __name__ == "__main__":
    cdnod_algo = AcceleratedCDNOD({})
    cdnod_algo.test_algorithm()
